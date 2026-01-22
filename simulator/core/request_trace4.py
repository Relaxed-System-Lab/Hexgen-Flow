import enum
from dataclasses import dataclass
from typing import Dict, List, Optional

from .new_obtain_latency import build_latency_dict as build_latency_dict_trace4


class REQ_STATUS(enum.Enum):
    PENDING = 1
    SCHEDULED = 2
    PREFILL = 3
    GENERATE = 4
    EXIT = 5


# Simplified workflow for trace4
STANDARD_WORKFLOW = [
    "Selector",
    "Decomposer",
    "Refiner",
    "Refiner",
    "Refiner",
    "Refiner",
]


# I/O length per step for trace4
EMPIRICAL_IO_LEN = {
    "Selector": (2962, 143),
    "Decomposer": (2310, 215),
    "Refiner": (1729, 555),
}

# Lookup build_latency_dict per hardware for trace4
LATENCY_DICT_LOOKUP = {
    "nvidia_A100": build_latency_dict_trace4("nvidia_A100"),
    "nvidia_A6000": build_latency_dict_trace4("nvidia_A6000"),
    "nvidia_L40S": build_latency_dict_trace4("nvidia_L40S"),
}


def calculate_empirical_time_by_io(hardware_name: str, input_len: int, output_len: int) -> float:
    latency_dict = LATENCY_DICT_LOOKUP[hardware_name]
    return latency_dict[(input_len, output_len)]["latency"]


def calculate_empirical_time(hardware_name: str, step_name: str) -> float:
    """Empirical time by step using default IO lengths for trace4."""
    input_len, output_len = EMPIRICAL_IO_LEN[step_name]
    return calculate_empirical_time_by_io(hardware_name, input_len, output_len)


def calculate_avg_empirical_time(hardware_lst: List[str], input_len: int, output_len: int) -> float:
    total_time = 0.0
    for hardware_name in hardware_lst:
        total_time += calculate_empirical_time_by_io(hardware_name, input_len, output_len)
    return total_time / max(len(hardware_lst), 1)


@dataclass
class GenerationRequest:
    req_id: str
    model: str
    step: str
    input_length: int
    output_length: int
    arrive_at: float
    SLO: float
    parent_request: Optional["Text2SQLRequest"] = None

    # runtime fields
    status: REQ_STATUS = REQ_STATUS.PENDING
    prefill_time: Optional[float] = None
    generated_tokens: int = 0
    prefill_finished_at: Optional[float] = None
    generation_finished_at: Optional[float] = None
    elapsed_time: float = 0.0
    time_left: float = 0.0
    urgency: float = 0.0

    def set_generation_finished_at(self, finished_at: float):
        self.generation_finished_at = finished_at

    def set_prefill_finished_at(self, finished_at: float):
        self.prefill_finished_at = finished_at

    def _prefill(self):
        self.status = REQ_STATUS.PREFILL

    def _decode(self) -> bool:
        self.generated_tokens += 1
        if self.generated_tokens == self.output_length:
            self._stop()
            return True
        return False

    def _stop(self):
        self.status = REQ_STATUS.EXIT

    def calculate_empirical_time(self, hardware_name: str) -> float:
        return calculate_empirical_time_by_io(hardware_name, self.input_length, self.output_length)

    def update_urgency(self, hardware_name: str):
        empirical_time = self.calculate_empirical_time(hardware_name)
        self.urgency = -((self.SLO - self.elapsed_time) - empirical_time)

    def to_dict(self) -> Dict:
        return {
            "req_id": self.req_id,
            "model": self.model,
            "step": self.step,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "arrive_at": self.arrive_at,
            "prefill_time": self.prefill_time,
            "generated_tokens": self.generated_tokens,
            "prefill_finished_at": self.prefill_finished_at,
            "generation_finished_at": self.generation_finished_at,
        }


@dataclass
class Text2SQLRequest:
    req_id: str
    gen_requests_config: List[dict]
    slo: float = 0.0
    hardware_lst: List[str] = None
    tenant_id: int = 0
    arrive_at: float = 0.0
    aborted: bool = False

    # runtime fields
    current_stage: int = 0
    total_stages: int = 0
    total_time: float = 0.0
    current_requests: Optional[List[GenerationRequest]] = None
    stage_lst: List[str] = None
    request_counter: int = 0

    def __init__(
        self,
        req_id: str,
        gen_requests_config: List[dict],
        slo: float = 0.0,
        hardware_lst: List[str] = None,
        tenant_id: int = 0,
        arrive_at: float = 0.0,
    ):
        self.slo = slo
        self.req_id = req_id
        self.current_stage = 0
        self.stage_lst = []
        self.gen_requests_config = gen_requests_config
        self.total_time = 0.0
        self.current_requests = []
        self.request_counter = 0
        self.tenant_id = tenant_id
        self.arrive_at = arrive_at
        self.aborted = False
        self._initialize_stages()
        self.total_stages = len(self.stage_lst)
        # self.standard_workflow = STANDARD_WORKFLOW.copy()
        self.hardware_lst = hardware_lst or ["nvidia_A100"]

    def _initialize_stages(self):
        for stage_config in self.gen_requests_config:
            self.stage_lst.append(stage_config["step"])
        self.standard_workflow = self.stage_lst.copy()

    def create_current_stage_requests(self, model, arrive_at: float) -> List[Optional[GenerationRequest]]:
        current_step = self.stage_lst[self.current_stage]
        if current_step in self.standard_workflow:
            self.standard_workflow.remove(current_step)
        # if current_step == "Decomposer" and "Selector" in self.standard_workflow:
        #     self.standard_workflow.remove("Selector")

        # Calculate per-step SLO proportionally by average empirical time using provided input/output lengths
        # We look up the first config that matches the current step
        # Prefer provided lengths; fall back to step defaults
        input_len = None
        output_len = None
        # for stage_config in self.gen_requests_config:
        #     if stage_config["step"] == current_step:
        #         input_len = stage_config.get("input_length")
        #         output_len = stage_config.get("output_length")
        #         break
        if input_len is None or output_len is None:
            input_len, output_len = EMPIRICAL_IO_LEN[current_step]

        cur_avg = calculate_avg_empirical_time(self.hardware_lst, input_len, output_len)

        # Remaining workflow time estimate
        remaining_est = 0.0
        # Estimate remaining steps using their configured input/output from gen_requests_config order
        for s in self.standard_workflow:
            il, ol = EMPIRICAL_IO_LEN[s]
            remaining_est += calculate_avg_empirical_time(self.hardware_lst, il, ol)

        denom = cur_avg + remaining_est
        step_slo = (self.slo - self.total_time) * (cur_avg / denom)

        next_request = GenerationRequest(
            req_id=f"{self.req_id}_req_{self.request_counter}",
            model=model,
            step=current_step,
            SLO=step_slo,
            input_length=input_len,
            output_length=output_len,
            arrive_at=arrive_at,
            parent_request=self,
        )
        self.current_requests.append(next_request)
        self.request_counter += 1
        return self.current_requests

    def update_stage(self, request: GenerationRequest, current_time: float):
        self.current_requests.remove(request)
        if self.current_requests == []:
            self.total_time += (current_time - request.arrive_at)
            self.current_stage += 1
