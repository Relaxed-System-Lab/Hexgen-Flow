from ._base import BaseGlobalToLocalPolicy
from simulator.core.request import GenerationRequest
from simulator.core.engine_optimized import LLMEngineOptimized
from typing import Dict, List
import itertools
import json

PRICE_PER_HOUR_H100 = 2.822
PRICE_PER_HOUR_A40 = 0.6
PRICE_PER_HOUR_A100 = 1.4
PRICE_PER_HOUR_A6000 = 0.5

# def cost(hardware_name: str, step: str) -> float:
#     if hardware_name == "nvidia_H100":
#         return PRICE_PER_HOUR_H100 * calculate_empirical_time(hardware_name, step)
#     elif hardware_name == "nvidia_A40":
#         return PRICE_PER_HOUR_A40 * calculate_empirical_time(hardware_name, step)
#     elif hardware_name == "nvidia_A100":
#         return PRICE_PER_HOUR_A100 * calculate_empirical_time(hardware_name, step)
#     elif hardware_name == "nvidia_A6000":
#         return PRICE_PER_HOUR_A6000 * calculate_empirical_time(hardware_name, step)
#     else:
#         raise ValueError(f"Unknown hardware name: {hardware_name}")

class WorkloadBalancePolicy(BaseGlobalToLocalPolicy):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.engines = None
        self.baseline_engine = None
        self.engine_queue = {}
        self.beta = float(beta)
        # Default to the standard request.calculate_empirical_time; can be overridden for trace4
        try:
            from simulator.core.request import calculate_empirical_time as _default_calc
            self._calculate_empirical_time = _default_calc
        except Exception:
            self._calculate_empirical_time = lambda hw, step: 0.0

    def set_empirical_time_fn(self, fn):
        self._calculate_empirical_time = fn

    def prepare(self, engines: Dict[str, List[LLMEngineOptimized]]):
        self.engines = engines
        for model, engine_list in engines.items():
            for engine in engine_list:
                self.engine_queue[engine.engine_id] = []
        self.baseline_engine = self.engines["meta-llama/Llama-3.1-70B-Instruct"][0]

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def _engine_workload(self, engine: LLMEngineOptimized) -> float:
        workload = 0.0
        for request in itertools.chain(engine.waiting, engine.running):
            workload += self._calculate_empirical_time(engine.hardware_name, request.step)
        return workload

    def _update_json_file(self, engine_name: str, step_name: str, file_path: str = "engine_queues.json"):
        """Update the JSON file with the step_name for the given engine."""
        try:
            # Load existing data from the JSON file
            with open(file_path, "r") as json_file:
                engine_queues = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize if the file doesn't exist or is invalid
            engine_queues = {}

        # Update the engine's queue in the JSON data
        if engine_name not in engine_queues:
            engine_queues[engine_name] = []
        engine_queues[engine_name].append(step_name)

        # Write the updated data back to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(engine_queues, json_file, indent=4)

    def assign_requests(self, request: GenerationRequest, alpha: float = 0.0):
        baseline_workload = self._engine_workload(self.baseline_engine)
        if baseline_workload == 0:
            # self._update_json_file(self.baseline_engine.engine_id, request.step)  # Update JSON
            self.baseline_engine.add_request(request)
            self.engine_queue[self.baseline_engine.engine_id].append(request.step)
            return
        highest_priority = float("-inf")
        wanted_engine = None     
        for e in self.engines[request.model]:
            engine_workload = self._engine_workload(e)
            denom = self.beta * baseline_workload
            if denom == 0:
                denom = baseline_workload if baseline_workload != 0 else 1.0
            if e.hardware_name == "nvidia_A6000":
                workload_ratio = engine_workload * 1 / denom
            elif e.hardware_name == "nvidia_L40S":
                workload_ratio = engine_workload * 1 / denom
            else:
                workload_ratio = engine_workload / denom
            # cost_ratio based on empirical time per step
            base_est = self._calculate_empirical_time(self.baseline_engine.hardware_name, request.step)
            e_est = self._calculate_empirical_time(e.hardware_name, request.step)
            cost_ratio = base_est / e_est if e_est != 0 else float("inf")
            priority = alpha * cost_ratio - (1-alpha) * workload_ratio
            if priority > highest_priority:
                highest_priority = priority
                wanted_engine = e
        if wanted_engine:
            # self._update_json_file(wanted_engine.engine_id, request.step)  # Update JSON
            wanted_engine.add_request(request)
            self.engine_queue[wanted_engine.engine_id].append(request.step)
