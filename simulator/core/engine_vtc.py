from copy import deepcopy
from collections import deque
from simulator.internal.analyzer import ModelAnalyzer
from simulator.core.trace import TraceEvent
from simulator.core.memory_planner import MemoryPlanner
from simulator.internal.configs.hardware_params import hardware_params
from typing import List, Deque, Optional
import uuid
import numpy as np

from .request import GenerationRequest
from .global_waitlist import GlobalWaitlist  # Import global waitlist
from .obtain_latency import build_latency_dict


class LLMEngineVTC:
    def __init__(self, engine_id, model_name, hardware_name, w_bit, a_bit, kv_bit):
        self.input_price = 1
        self.output_price = 2
        self.served = {}
        self.user_req_list = {}
        self.fairw = {}
    
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
        self.waiting: Deque[GenerationRequest] = deque()
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

    def add_request(self, req: GenerationRequest):
        self.waiting.append(req)
        parent_id = req.parent_request.req_id if req.parent_request is not None else req.req_id
        if parent_id not in self.user_req_list:
            self.user_req_list[parent_id] = deque([req])
            self.served[parent_id] = 0
            self.fairw[parent_id] = 1
        else:
            self.user_req_list[parent_id].append(req)

        # waiting queue was empty before
        if len(self.user_req_list[parent_id]) == 1:
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (len(self.user_req_list[k]) > 0 and k != parent_id)]
            if len(cnts) > 0:
                self.served[parent_id] = max(self.served[parent_id], min(cnts))

    def update_elapsed_time(self, current_time: float, hardware_name: str):
        for request in self.waiting:
            request.elapsed_time = current_time - request.arrive_at
            request.update_urgency(hardware_name)

    def get_highest_priority_request(self, waitlist) -> Optional[GenerationRequest]:
        if not waitlist:
            return None
        if len(self.served) == 0:
            return None
        active_served = {k: v for k, v in self.served.items()}
        # Find the currently waiting request with the lowest counter (self.served)
        while True:
            if len(active_served) == 0:
                break
            parent_request_id = min(active_served, key=active_served.get)
            if len(self.user_req_list[parent_request_id]) > 0:
                req = self.user_req_list[parent_request_id][0]
                if self.memory_planner.can_allocate_request(req):
                    self.user_req_list[parent_request_id].popleft()
                    # update fairness counter
                    self.served[parent_request_id] += req.input_length * self.input_price / self.fairw[parent_request_id]
                    active_served[parent_request_id] += req.input_length * self.input_price / self.fairw[parent_request_id]
                    return req
                else:
                    break
            else:
                del active_served[parent_request_id]
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
        # Fetch the highest-priority request from the global waitlist
        self.update_elapsed_time(start_at, self.hardware_name)
        next_request = self.get_highest_priority_request(self.waiting)

        if next_request:
            self.waiting.remove(next_request)
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
