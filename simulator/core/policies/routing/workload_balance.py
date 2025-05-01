from ._base import BaseGlobalToLocalPolicy
from simulator.core.request import GenerationRequest, calculate_empirical_time
from simulator.core.engine_optimized import LLMEngine
from typing import Dict, List
import itertools
import json

PRICE_PER_HOUR_H100 = 2.822
PRICE_PER_HOUR_A40 = 0.6
PRICE_PER_HOUR_A100 = 1.4
PRICE_PER_HOUR_A6000 = 0.5

def cost(hardware_name: str, step: str) -> float:
    if hardware_name == "nvidia_H100":
        return PRICE_PER_HOUR_H100 * calculate_empirical_time(hardware_name, step)
    elif hardware_name == "nvidia_A40":
        return PRICE_PER_HOUR_A40 * calculate_empirical_time(hardware_name, step)
    elif hardware_name == "nvidia_A100":
        return PRICE_PER_HOUR_A100 * calculate_empirical_time(hardware_name, step)
    elif hardware_name == "nvidia_A6000":
        return PRICE_PER_HOUR_A6000 * calculate_empirical_time(hardware_name, step)
    else:
        raise ValueError(f"Unknown hardware name: {hardware_name}")

class WorkloadBalancePolicy(BaseGlobalToLocalPolicy):
    def __init__(self):
        super().__init__()
        self.engines = None
        self.engine_workloads = {}
        self.baseline_engine = None
        self.engine_queue = {}

    def prepare(self, engines: Dict[str, List[LLMEngine]]):
        self.engines = engines
        for model, engine_list in engines.items():
            for engine in engine_list:
                self.engine_workloads[engine] = 0
                self.engine_queue[engine.engine_id] = []
        self.baseline_engine = self.engines["meta-llama/Llama-3.1-70B-Instruct"][0]

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
        if self.engine_workloads[self.baseline_engine] == 0:
            # self._update_json_file(self.baseline_engine.engine_id, request.step)  # Update JSON
            self.baseline_engine.add_request(request)
            self.engine_workloads[self.baseline_engine] += calculate_empirical_time(self.baseline_engine.hardware_name, request.step)
            # self.engine_workloads[self.baseline_engine] += self.baseline_engine.latency_dict[(request.input_length, request.output_length)]["latency"]
            self.engine_queue[self.baseline_engine.engine_id].append(request.step)
            return
        highest_priority = float("-inf")
        wanted_engine = None     
        for e in self.engines[request.model]:
            if e.hardware_name == "nvidia_A6000":
                workload_ratio = self.engine_workloads[e]*0.5 / self.engine_workloads[self.baseline_engine]
            elif e.hardware_name == "nvidia_L40S":
                workload_ratio = self.engine_workloads[e]*0.5 / self.engine_workloads[self.baseline_engine]
            else:
                workload_ratio = self.engine_workloads[e] / self.engine_workloads[self.baseline_engine]
            # cost_ratio = cost(self.baseline_engine.hardware_name, request.step) / cost(e.hardware_name, request.step)
            cost_ratio = calculate_empirical_time(self.baseline_engine.hardware_name, request.step) / calculate_empirical_time(e.hardware_name, request.step)
            priority = alpha * cost_ratio - (1-alpha) * workload_ratio
            if priority > highest_priority:
                highest_priority = priority
                wanted_engine = e
        if wanted_engine:
            # self._update_json_file(wanted_engine.engine_id, request.step)  # Update JSON
            wanted_engine.add_request(request)
            self.engine_workloads[wanted_engine] += calculate_empirical_time(wanted_engine.hardware_name, request.step)
            # self.engine_workloads[wanted_engine] += wanted_engine.latency_dict[(request.input_length, request.output_length)]["latency"]
            self.engine_queue[wanted_engine.engine_id].append(request.step)