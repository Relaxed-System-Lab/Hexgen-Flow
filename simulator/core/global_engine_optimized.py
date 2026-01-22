from collections import deque, defaultdict
import math
import os
import random
from typing import List, Deque
from simulator.core.engine_optimized import LLMEngineOptimized
from simulator.core.engine_vtc import LLMEngineVTC
from simulator.core.engine_qlm import LLMEngineQLM
from simulator.core.engine_sjf import LLMEngineSJF
from simulator.core.request import GenerationRequest
from dataclasses import dataclass
from .policies import EvenGTLPolicy
from .policies import WorkloadBalancePolicy
import json
import numpy as np
from typing import Dict, List, Optional
from simulator.core.request import Text2SQLRequest, GenerationRequest
from simulator.core.arrival import PoissonProcess
from simulator.core.arrival import DeterministicProcess
from simulator.core.global_waitlist import GlobalWaitlist  # Import global waitlist


class OPGlobalEngine:
    def __init__(self, alpha, mode):
        self.alpha = alpha
        self.hardware_lst = []
        self._is_trace4 = False
        if mode == "hexflow":
            self.engines = defaultdict(list[LLMEngineOptimized])
        elif mode == "vtc":
            self.engines = defaultdict(list[LLMEngineVTC])
        elif mode == "qlm":
            self.engines = defaultdict(list[LLMEngineQLM])
        elif mode == "sjf":
            self.engines = defaultdict(list[LLMEngineSJF])
        self.mode = mode
        self.timers = defaultdict(dict)
        self.pending_requests: Deque[GenerationRequest] = deque()
        self.global_timer = 0
        self.supported_models: List[str] = []
        self._trace = []
        self.total_requests = 0
        self.policy = WorkloadBalancePolicy()
        self.text2sql_requests: Dict[str, Text2SQLRequest] = {}
        self.background_requests: Dict[str, GenerationRequest] = {}
        # self.served = {}
        # self.global_waitlist = GlobalWaitlist.get_instance()  # Initialize global waitlist

    def add_engine(self, w1, model_name, hardware_name, w_bit, a_bit, kv_bit):
        existing_engines = sum([len(x) for x in self.engines.values()])
        if self.mode == "hexflow":
            engine = LLMEngineOptimized(
                w1, existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
            )
        elif self.mode == "vtc":
            engine = LLMEngineVTC(
                existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
            )
        elif self.mode == "qlm":
            engine = LLMEngineQLM(
                existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
            )
        elif self.mode == "sjf":
            engine = LLMEngineSJF(
                existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
            )
        # If this is a trace4 workflow, override latency dict using new_obtain_latency
        if getattr(self, "_is_trace4", False):
            try:
                from simulator.core.new_obtain_latency import build_latency_dict as _build_trace4
                engine.latency_dict = _build_trace4(hardware_name)
            except Exception:
                pass
        self.engines[model_name].append(engine)
        self.hardware_lst.append(hardware_name)
        if model_name not in self.supported_models:
            self.supported_models.append(model_name)
        self.policy.prepare(self.engines)
        self.timers[model_name][engine.engine_id] = 0

    def load_requests(
        self,
        input_file: str,
        arrival_rate: float,
        slo: float,
        multi_tenant: bool = False,
        arrival_shift: bool = False,
    ):
        with open(input_file, 'r') as f:
            data = json.load(f)
        base_name = os.path.basename(input_file)
        background_ratio = None
        if base_name.startswith("input_file_trace5"):
            background_ratio = 0.25
        elif base_name.startswith("input_file_trace6"):
            background_ratio = 0.5
        # Detect trace4 workflow: steps are subset of {Selector, Decomposer, Refiner}
        trace4_steps = {"Selector", "Decomposer", "Refiner"}
        def _is_trace4_data(dlist):
            try:
                for item in dlist:
                    steps = [cfg.get("step") for cfg in item.get("Text2SQLRequest", [])]
                    if not steps:
                        return False
                    if not set(steps).issubset(trace4_steps):
                        return False
                return True
            except Exception:
                return False
        self._is_trace4 = _is_trace4_data(data)
        if self._is_trace4:
            # Update latency dicts for already-created engines.
            try:
                from simulator.core.new_obtain_latency import build_latency_dict as _build_trace4
                for model in self.supported_models:
                    for engine in self.engines[model]:
                        engine.latency_dict = _build_trace4(engine.hardware_name)
            except Exception:
                pass
        if background_ratio is not None:
            background_count = int(math.ceil(len(data) * background_ratio / (1 - background_ratio)))
            candidates = []
            for item in data:
                for cfg in item.get("Text2SQLRequest", []):
                    if "input_length" in cfg and "output_length" in cfg:
                        candidates.append((cfg["input_length"], cfg["output_length"]))
            rng = random.Random(0)
            background_requests = []
            for i in range(background_count):
                if candidates:
                    input_len, output_len = rng.choice(candidates)
                else:
                    input_len, output_len = 1, 1
                background_requests.append(
                    {
                        "req_id": f"background_{i}",
                        "step": f"background:{input_len}:{output_len}",
                        "input_length": input_len,
                        "output_length": output_len,
                    }
                )
            combined = []
            for idx, request_data in enumerate(data):
                combined.append(("text2sql", idx, request_data))
            for i, bg_req in enumerate(background_requests):
                combined.append(("background", i, bg_req))
            rng.shuffle(combined)
            total_count = len(combined)
        else:
            combined = [("text2sql", idx, request_data) for idx, request_data in enumerate(data)]
            total_count = len(data)

        if arrival_rate is not None and arrival_rate > 0:
            if arrival_shift:
                # Piecewise workload shift: 3s @ 5 qps, 4s @ 20 qps, 3s @ 5 qps
                print("Synthesizing shifted arrival process: 3s@5qps, 4s@20qps, 3s@5qps")
                segments = [
                    (0.0, 3.0, 5.0),
                    (3.0, 7.0, 20.0),
                    (7.0, 10.0, 5.0),
                ]
                counts = [int(round((end - start) * rate)) for start, end, rate in segments]
                diff = total_count - sum(counts)
                if diff != 0:
                    counts[-1] = max(counts[-1] + diff, 0)
                workload = []
                for (start, end, _rate), count in zip(segments, counts):
                    if count <= 0:
                        continue
                    interval = (end - start) / count
                    workload.extend([start + i * interval for i in range(count)])
                # Trim or pad to match total_count exactly.
                if len(workload) > total_count:
                    workload = workload[:total_count]
                elif len(workload) < total_count:
                    workload.extend([segments[-1][1]] * (total_count - len(workload)))
            else:
                print(f"Synthesizing arrival time using Poisson process with ar {arrival_rate}")
                pp = PoissonProcess(arrival_rate)
                # print(f"Synthesizing arrival time using Deterministic process with ar {arrival_rate}")
                # pp = DeterministicProcess(arrival_rate)
                duration_needed = total_count / arrival_rate
                workload = pp.generate_workload(
                    0, duration_needed * 2
                )  # multiply by 2 to be safe
        else:
            print("Arrival rate not provided, assuming all requests arrive at time 0")
            workload = [0] * total_count
        
        # Calculate tenant-specific SLOs
        if multi_tenant:
            from simulator.core.request import calculate_tenant_slo
            tenant1_slo = calculate_tenant_slo(self.hardware_lst, 0)  # 2.0x SLO
            tenant2_slo = calculate_tenant_slo(self.hardware_lst, 1)  # Base SLO
            print(f"Multi-tenant mode: Tenant 1 SLO = {tenant1_slo:.2f}s, Tenant 2 SLO = {tenant2_slo:.2f}s")
        else:
            tenant1_slo = slo
            tenant2_slo = slo
        
        # Choose request class and empirical time fn based on detected workflow
        if self._is_trace4:
            from simulator.core.request_trace4 import Text2SQLRequest as _RequestCls
            from simulator.core.request_trace4 import calculate_empirical_time as _calc_empirical
        else:
            from simulator.core.request import Text2SQLRequest as _RequestCls
            from simulator.core.request import calculate_empirical_time as _calc_empirical
        # Set policy empirical time function
        self.policy.set_empirical_time_fn(_calc_empirical)

        for i, item in enumerate(combined):
            kind, idx, request_data = item
            arrive_at = workload[i]
            if kind == "text2sql":
                request_data["model"] = "meta-llama/Llama-3.1-70B-Instruct"

                # Assign tenant based on request order (first half = tenant 1, second half = tenant 2)
                if multi_tenant:
                    tenant_id = 0 if idx < len(data) // 2 else 1
                    tenant_slo = tenant1_slo if tenant_id == 0 else tenant2_slo
                else:
                    tenant_id = 0
                    tenant_slo = slo

                text2sql_req = _RequestCls(
                    req_id=f"text2sql_{idx}",
                    gen_requests_config=request_data["Text2SQLRequest"],
                    slo=tenant_slo,
                    hardware_lst=self.hardware_lst,
                    tenant_id=tenant_id,
                    arrive_at=arrive_at,
                )
                self.text2sql_requests[text2sql_req.req_id] = text2sql_req
                first_requests = text2sql_req.create_current_stage_requests(
                    request_data["model"], arrive_at
                )
                for req in first_requests:
                    if req is not None:
                        self.pending_requests.append(req)
                        self.total_requests += 1
            else:
                bg_req = GenerationRequest(
                    req_id=request_data["req_id"],
                    model="meta-llama/Llama-3.1-70B-Instruct",
                    step=request_data["step"],
                    slo=slo,
                    input_length=request_data["input_length"],
                    output_length=request_data["output_length"],
                    arrive_at=arrive_at,
                    parent_request=None,
                )
                self.background_requests[bg_req.req_id] = bg_req
                self.pending_requests.append(bg_req)
                self.total_requests += 1
    
    def handle_request_completion(self, request: GenerationRequest, current_time: float):
        if request.parent_request:
            parent = request.parent_request
            parent.update_stage(request, current_time)
            if parent.current_requests == [] and parent.current_stage < parent.total_stages:
                next_requests = parent.create_current_stage_requests("meta-llama/Llama-3.1-70B-Instruct", current_time)
                for next_request in next_requests:
                    if next_request is not None:
                        self.pending_requests.append(next_request)
            if parent.current_stage >= parent.total_stages:
                return parent
        return None
    
    def SLO_pass_rate(self, SLO):
        pass_rate = 0
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                continue
            else:
                if request.total_time <= SLO:
                    pass_rate += 1
        return pass_rate / len(self.text2sql_requests)
    
    def tenant_SLO_pass_rate(self, tenant_id: int, SLO: float):
        """Calculate SLO pass rate for a specific tenant"""
        tenant_requests = [req for req in self.text2sql_requests.values() if req.tenant_id == tenant_id]
        if not tenant_requests:
            return 0.0
        
        pass_rate = 0
        for request in tenant_requests:
            if request.current_stage < request.total_stages:
                continue
            else:
                if request.total_time <= SLO:
                    pass_rate += 1
        return pass_rate / len(tenant_requests)
    
    def multi_tenant_SLO_pass_rate(self, tenant1_SLO: float, tenant2_SLO: float):
        """Calculate SLO pass rates for both tenants"""
        tenant1_pass_rate = self.tenant_SLO_pass_rate(0, tenant1_SLO)
        tenant2_pass_rate = self.tenant_SLO_pass_rate(1, tenant2_SLO)
        return tenant1_pass_rate, tenant2_pass_rate

    def save_results(self, output_file: str):
        results = {}
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                results.update({req_id: -1})
            else:
                results.update({req_id: request.total_time})
        for req_id, request in self.background_requests.items():
            if request.generation_finished_at is None:
                results.update({req_id: -1})
            else:
                results.update({req_id: request.generation_finished_at - request.arrive_at})
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    def start(self):
        print(f"Total requests: {self.total_requests}")
        time_queue = set()
        last_report_time = -1.0
        while True:
            for model in self.supported_models:
                for engine in self.engines[model]:
                    if self.timers[model][engine.engine_id] <= self.global_timer:
                        event, finished_lst, next_time, memory_event = engine.step(
                            self.timers[model][engine.engine_id]
                        )
                        self.timers[model][engine.engine_id] = next_time
                        if finished_lst:
                            for req in finished_lst:
                                self.handle_request_completion(req, next_time)
                        time_queue.add(next_time)
                        if event is not None:
                            self._trace.extend(event)
                            self._trace.append(memory_event)
            if time_queue:
                self.global_timer = min(time_queue)
                time_queue.remove(self.global_timer)
                self.check_new_requests(self.global_timer)

            # Progress printing disabled to reduce stdout overhead.
            if self.global_timer - last_report_time >= 1.0:
                last_report_time = self.global_timer
                # print(f"Finished: {self.finished_percentage:.2f}%, Current Time: {self.global_timer:.2f}", end="\r",)
            if not self.has_remaining_requests():
                # save the engine queue to a JSON file
                # with open("engine_queues.json", "w") as f:
                    # json.dump(self.policy.engine_queue, f, indent=4)
                break

    def has_remaining_requests(self):
        if self.pending_requests:
            return True
        for model in self.supported_models:
            for engine in self.engines[model]:
                if len(engine.waiting) > 0 or len(engine.running) > 0:
                    return True
        return False

    def check_new_requests(self, end_at):
        if len(self.pending_requests) > 0:
            allocatable_requests = [x for x in self.pending_requests if x.arrive_at <= end_at]
            for req in allocatable_requests:
                # self.global_waitlist.add_request(req)  # Add to global waitlist
                self.policy.assign_requests(req, self.alpha)
                self.pending_requests.remove(req)

    @property
    def finished_percentage(self):
        total_finished = 0
        for model in self.supported_models:
            for engine in self.engines[model]:
                total_finished += engine.finished_requests
        return 100 * total_finished / self.total_requests

    @property
    def trace(self):
        return self._trace

    @property
    def requests_stats(self):
        stats = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                stats.extend([x.to_dict() for x in engine.finished])
        return stats

    @property
    def config(self):
        configuration = {}
        for model in self.supported_models:
            configuration[model] = {}
            engines = []
            for engine in self.engines[model]:
                engine_config = {
                    'model': model,
                    'hardware': engine.hardware_name,
                    'w_bit': engine.w_bit,
                    'a_bit': engine.a_bit,
                    'kv_bit': engine.kv_bit,
                }
                engines.append(engine_config)
            configuration[model]['engines'] = engines
        return configuration            
        

    @property
    def summary(self):
        stats = self.requests_stats

        avg_latency = sum(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        throughput = len(stats) / max([x["generation_finished_at"] for x in stats])
        p90_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.9)]
        p95_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.95)]
        avg_time_to_first_token = sum(
            [x["prefill_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        p90_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.9)
        ]
        p95_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.95)
        ]
        summaries = [
            {"Metric": "Avg E2E-Latency (s)", "Value": avg_latency},
            {"Metric": "Avg TTFT (s)", "Value": avg_time_to_first_token},
            {"Metric": "Throughput (req/s)", "Value": throughput},
            {"Metric": "P90 Latency (s)", "Value": p90_latency},
            {"Metric": "P95 Latency (s)", "Value": p95_latency},
            {"Metric": "P90 TTFT (s)", "Value": p90_ttft},
            {"Metric": "P95 TTFT (s)", "Value": p95_ttft},
        ]
        return summaries

    @property
    def failed_requests(self):
        failed_requests = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                failed_requests.extend([x.to_dict() for x in engine.failed])
        return failed_requests
