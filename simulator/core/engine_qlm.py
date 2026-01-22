from copy import deepcopy
from collections import deque
from simulator.internal.analyzer import ModelAnalyzer
from simulator.core.trace import TraceEvent
from simulator.core.memory_planner import MemoryPlanner
from simulator.internal.configs.hardware_params import hardware_params
from typing import List, Deque, Optional

from .request import GenerationRequest
from .global_waitlist import GlobalWaitlist  # Import global waitlist
from .obtain_latency import build_latency_dict

import uuid
from collections import deque
from .rwt_estimator import RWTEstimator
from simulator.core.request import STANDARD_WORKFLOW, calculate_avg_empirical_time

hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A6000", "nvidia_L40S"]
# Calculate the average instead of the sum
# SLO = sum([calculate_avg_empirical_time(hardware_lst, s) for s in STANDARD_WORKFLOW]) / len(STANDARD_WORKFLOW)

class Group:
    """
    Group class is used to store the requests that are in the same request group.
    Request group is a group of requests that have the same model and similar clustered SLO.
    """

    def __init__(self, model, time_left):
        self.group_id = uuid.uuid4()
        self.model = model
        self.time_left = time_left
        self.requests = deque()

    def add_request(self, request):
        self.requests.append(request)

    def pop_request(self):
        return self.requests.popleft()

    def __hash__(self):
        return hash(self.group_id)


class VirtualQueue:
    """
    A VirtualQueue is a queue that contains a list of request groups. Each request group is a list of requests.
    """

    def __init__(self):
        self.vq_id = uuid.uuid4()
        self.groups = deque()

    def add_group(self, group):
        self.groups.append(group)

    def pop_group(self):
        return self.groups.popleft()

    def get_head_group(self):
        return self.groups[0]

    def __hash__(self):
        return hash(self.vq_id)
    
    def __len__(self):
        return sum(len(group.requests) for group in self.groups)


class LLMEngineQLM:
    def __init__(self, engine_id, model_name, hardware_name, w_bit, a_bit, kv_bit):
        self.rwt_estimator = RWTEstimator()
        self.engine_id = engine_id
        self.model_name = model_name
        self.hardware_name = hardware_name
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.analyzer = ModelAnalyzer(
            model_id=model_name,
            hardware=hardware_name,
            config_file="simulator/internal/configs/llama.py",
            source="huggingface",
        )
        self.global_waitlist = GlobalWaitlist.get_instance()
        self.waiting = VirtualQueue()
        self.running: Deque[GenerationRequest] = deque()
        self.finished: List[GenerationRequest] = []
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            hardware_params[hardware_name],
            w_bit,
            a_bit,
            kv_bit,
        )
        self.memory_planner.print_status()
        self.latency_dict = build_latency_dict(hardware_name)
        self.finished_requests: int = 0
        self.configure()

    def configure(self):
        pass

    def _update_all_slos(self):
        """
        Updates the SLOs for all requests and their groups based on current time.
        :param vqs: The list of virtual queues.
        """
        for group in self.waiting.groups:
            min_time_left = float('inf')
            for request in group.requests:
                if request.parent_request is None:
                    slo = request.SLO
                else:
                    slo = request.parent_request.slo / len(STANDARD_WORKFLOW)
                request.time_left = (slo - request.elapsed_time) // 10 * 10
                min_time_left = min(min_time_left, request.time_left)
            group.time_left = min_time_left  # Update group's time_left to match its most urgent request

    def check_violation(self):
        """
        Checks for SLO violations in the virtual queues.
        :param vqs: The list of virtual queues.
        :return: True if there is a violation, False otherwise.
        """
        self._update_all_slos()
        est_time = 0

        for group in self.waiting.groups:
            waiting_time = self.rwt_estimator.get_waiting_time(group)
            est_time += waiting_time
            if est_time > group.time_left:
                return True
        return False
    
    def _reorder_edf(self):
        """
        Reorders the virtual queues based on the Earliest Deadline First (EDF) policy.
        :param vqs: The list of virtual queues.
        :return: The reordered list of virtual queues.
        """

        groups = list(self.waiting.groups)
        groups.sort(key=lambda x: x.time_left)
        self.waiting.groups = deque(groups)

    def add_request(self, request: GenerationRequest):
        # Calculate request's time_left
        if request.parent_request is None:
            slo = request.SLO
        else:
            slo = request.parent_request.slo / len(STANDARD_WORKFLOW)
        request.time_left = (slo - request.elapsed_time) // 10 * 10
        
        # Find group with matching model and similar time_left
        target_group = None
        for group in self.waiting.groups:
            if group.model == request.model:
                # Add to group if time_left is within same bucket (rounded to nearest 10)
                if (group.time_left // 10) == (request.time_left // 10):
                    target_group = group
                    break
                    
        if target_group is None:
            # Create new group if no matching group found
            target_group = Group(request.model, request.time_left)
            self.waiting.add_group(target_group)
            
        # Insert request into group's deque (deque maintains FIFO order automatically)
        target_group.add_request(request)
        target_group.time_left = min(target_group.time_left, request.time_left)
        
        # Check violation and reorder groups if needed
        if self.check_violation():
            self._reorder_edf()

    def update_elapsed_time(self, current_time: float, hardware_name: str):
        for group in self.waiting.groups:
            for request in group.requests:
                request.elapsed_time = current_time - request.arrive_at
                request.update_urgency(hardware_name)

    def get_highest_priority_request(self, waitlist) -> Optional[GenerationRequest]:
        # 检查violation并排序
        if self.check_violation():
            self._reorder_edf()
        # 不再使用priority，直接取第一个group的第一个request
        if not waitlist.groups or not waitlist.groups[0].requests:
            return None
        wanted_request = waitlist.groups[0].requests[0]
        if self.memory_planner.can_allocate_request(wanted_request):
            return wanted_request
        return None

    def _prefill(self, request: GenerationRequest, start_at: float):
        self.memory_planner.allocate(request)
        memory_event = self.memory_event(start_at)
        if start_at < request.arrive_at:
            start_at = request.arrive_at
        self.running.append(request)
        request._prefill()
        prefill_time = self.latency_dict[(request.input_length, request.output_length)]["prefill_latency"]
        request.set_prefill_finished_at(start_at + prefill_time)
        if request.output_length == 1:
            request.set_generation_finished_at(start_at + prefill_time)
            self.memory_planner.free([request.req_id])
        return prefill_time + start_at, [request], memory_event

    def _decode(self, requests: List[GenerationRequest], start_at: float):
        max_batch_size = len(requests)
        decode_time = []
        finished_requests_in_this_batch = []
        executable_requests = []
        for req in requests:
            if self.memory_planner.can_allocate_request(req):
                self.memory_planner.allocate(req)
                executable_requests.append(req)
        batch_size = len(executable_requests)
        memory_event = self.memory_event(start_at)
        for req in executable_requests:
            if start_at < req.arrive_at:
                start_at = req.arrive_at
            decode_time.append(
                self.latency_dict[(req.input_length, req.output_length)]["per_token_decode_latency"]
            )
        finished_at = max(decode_time) + start_at
        finished_lst = []
        for req in executable_requests:
            finished = req._decode() #Check if the request is finished
            if finished:
                req.set_generation_finished_at(finished_at)
                self.finished_requests += 1
                self.running.remove(req)
                self.finished.append(req)
                finished_requests_in_this_batch.append(req.req_id)
                finished_lst.append(req)
        self.memory_planner.free(finished_requests_in_this_batch)
        return finished_at, executable_requests, memory_event, finished_lst

    def step(self, start_at: float):
        handled_requests = []
        # 更新所有request的elapsed_time
        self.update_elapsed_time(start_at, self.hardware_name)
        # 取第一个group的第一个request
        next_request = self.get_highest_priority_request(self.waiting)

        if next_request:
            # 从group中移除request，如果group空了则从virtual queue移除group
            for group in self.waiting.groups:
                if group.requests and group.requests[0] == next_request:
                    group.pop_request()
                    if not group.requests:
                        self.waiting.groups.remove(group)
                    break
            handled_requests = [next_request.req_id]
            prefill_end_at, handled_requests, memory_event = self._prefill(
                next_request, start_at
            )
            return (
                self.create_event(
                    "prefill", handled_requests, start_at, prefill_end_at
                ),
                [],
                prefill_end_at,
                memory_event,
            )
        elif len(self.running) > 0:
            decode_finished_at, handled_requests, memory_event, finished_lst = self._decode(
                list(self.running), start_at
            )
            return (
                self.create_event(
                    "decode", handled_requests, start_at, decode_finished_at
                ),
                finished_lst,
                decode_finished_at,
                memory_event,
            )
        else:
            return None, [], start_at + 0.0001, None

    def create_event(self, phase, handled_requests, start_at, end_at):
        complete_events = []
        handled_requests = [req.req_id for req in handled_requests]
        for req in handled_requests:
            complete = TraceEvent(
                name=f"{phase}-{req}",
                cat=f"{phase,req}",
                ph="X",
                pid=self.engine_id,
                tid=0,
                ts=int(start_at * 1000 * 1000),  # convert to microseconds
                dur=int((end_at - start_at) * 1000 * 1000),
            )
            complete_events.append(complete)
        return complete_events

    def memory_event(self, start_at):
        return TraceEvent(
            name="block usage",
            ph="C",
            ts=start_at * 1e6,
            pid=self.engine_id,
            tid=0,
            cat="memory",
            args={
                "used": self.memory_planner._allocated_blocks,
                "free": self.memory_planner._max_num_blocks
                - self.memory_planner._allocated_blocks,
            },
        )

    @property
    def empty(self):
        return len(self.waiting) == 0 and len(self.running) == 0
