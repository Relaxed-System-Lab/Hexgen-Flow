import enum
from datetime import datetime
from typing import Dict
from dataclasses import dataclass
from typing import List, Optional
from simulator.internal.configs.hardware_params import hardware_params
from .obtain_latency import build_latency_dict

class REQ_STATUS(enum.Enum):
    PENDING = 1
    SCHEDULED = 2
    PREFILL = 3
    GENERATE = 4
    EXIT = 5

# For model parameters: Llama-3.1-70B-Instruct
HIDDEN_SIZE = 8192
NUM_LAYERS = 80
B = 2

# EMPIRICAL_IO_LEN = {"Information Retriever": (306, 7),
#                   "extract_keywords": (669, 44), 
#                   "generate_candidate_llama-agent1": (10000, 680),
#                   "generate_candidate_llama-agent": (10000, 680),
#                   "revise": (5954, 82),
#                   "unit_tester": (275, 5),
#                   "generate_unit_test": (1110, 102),
#                   "evaluate": (960, 29)}
EMPIRICAL_IO_LEN = {"Information Retriever": (308, 5),
                  "extract_keywords": (667, 45), 
                  "generate_candidate_llama-agent1": (11748, 678),
                  "generate_candidate_llama-agent": (11748, 678),
                  "revise": (5963, 79),
                  "unit_tester": (275, 4),
                  "generate_unit_test": (1109, 103),
                  "evaluate": (961, 22)}

EMPIRICAL_TIME_A100 = {"Information Retriever": 0.2253,   # 0.2475
                        "extract_keywords": 1.1359,      #1.8238
                        "generate_candidate_llama-agent1": 18.2363,     #26.8
                        "generate_candidate_llama-agent": 18.2363,
                        "revise": 3.3314,        #3.1796
                        "unit_tester": 0.1735,      #0.209
                        "generate_unit_test": 2.5385,    #3.8955
                        "evaluate": 0.8670}     #0.9992
EMPIRICAL_TIME_H100 = {"Information Retriever": 0.1012,
                        "extract_keywords": 0.5469, 
                        "generate_candidate_llama-agent1": 8.7863,
                        "generate_candidate_llama-agent": 8.7863,
                        "revise": 1.4280,
                        "unit_tester": 0.0763,
                        "generate_unit_test": 1.2382,
                        "evaluate": 0.3985}
EMPIRICAL_TIME_A6000 = {"Information Retriever": 0.4869,     #0.3143
                        "extract_keywords": 2.1088,     #2.67
                        "generate_candidate_llama-agent1": 34.2747,    #41.32
                        "generate_candidate_llama-agent": 34.2747,
                        "revise": 8.0309,     #4.82
                        "unit_tester": 0.3907,    #0.254
                        "generate_unit_test": 4.5617,    #6.0753
                        "evaluate": 1.7923}    #1.324
EMPIRICAL_TIME_A40 = {"Information Retriever": 0.4932,
                        "extract_keywords": 2.5157,
                        "generate_candidate_llama-agent1": 40.3525,
                        "generate_candidate_llama-agent": 40.3525,
                        "revise": 13.8657,
                        "unit_tester": 0.3786,
                        "generate_unit_test": 5.6343,
                        "evaluate": 1.9049}
EMPIRICAL_TIME_L40 = {"Information Retriever": 0.4003,
                        "extract_keywords": 2.0330, 
                        "generate_candidate_llama-agent1": 32.6221,
                        "generate_candidate_llama-agent": 32.6221,
                        "revise": 5.8850,
                        "unit_tester": 0.3077,
                        "generate_unit_test": 4.5497,
                        "evaluate": 1.5440}
EMPIRICAL_TIME_A800 = {"Information Retriever": 0.1020,
                        "extract_keywords": 0.7857,
                        "generate_candidate_llama-agent1": 10.2678,
                        "generate_candidate_llama-agent": 10.2678,
                        "revise": 2.4632,
                        "unit_tester": 0.0970,
                        "generate_unit_test": 1.6258,
                        "evaluate": 0.8296}

LATENCY_DICT_LOOKUP = {
    "nvidia_A100": build_latency_dict("nvidia_A100"),
    "nvidia_A6000": build_latency_dict("nvidia_A6000"),
    "nvidia_L40S": build_latency_dict("nvidia_L40S")}


def calculate_empirical_time(hardware_name: str, step_name) -> float:
    """Calculate the empirical time for a given hardware"""
    # hardware_param = hardware_params[hardware_name]
    # c_t = int(hardware_param["FP16"])
    # m_t = int(hardware_param["bandwidth"])
    # input_length, output_length = EMPIRICAL_IO_LEN[step_name]
    # prefill = (24 * input_length * HIDDEN_SIZE**2 * NUM_LAYERS) / c_t
    # decode = (12 * output_length * HIDDEN_SIZE**2 * NUM_LAYERS * B) / m_t
    # empirical_time = prefill + decode
    latency_dict = LATENCY_DICT_LOOKUP[hardware_name]
    empirical_time = latency_dict[EMPIRICAL_IO_LEN[step_name]]["latency"]
    # if hardware_name == "nvidia_A100":
    #     empirical_time = EMPIRICAL_TIME_A100[step_name]
    return empirical_time

def calculate_avg_empirical_time(hardware_lst: List, step_name) -> float:
    """Calculate the average empirical time for a given hardware list"""
    total_time = 0
    for hardware_name in hardware_lst:
        total_time += calculate_empirical_time(hardware_name, step_name)
    return total_time / len(hardware_lst)

STANDARD_WORKFLOW = ["Information Retriever",
                     "extract_keywords",
                     "Information Retriever",
                     "Information Retriever",
                     "Information Retriever",
                     "generate_candidate_llama-agent",
                     "revise",
                     "revise",
                     "revise",
                     "revise",
                     "unit_tester",
                     "generate_unit_test",
                     "unit_tester",
                     "evaluate",
                     "unit_tester"]

class GenerationRequest:
    def __init__(self, req_id: str, model: str, step: str, slo: float, input_length: int, output_length: int, 
                 arrive_at: float, parent_request: Optional['Text2SQLRequest'] = None):
        self.req_id = req_id
        self.model = model
        self.step = step
        self.input_length = input_length
        self.output_length = output_length
        self.arrive_at = arrive_at
        self.status = REQ_STATUS.PENDING
        self.prefill_time = None
        self.parent_request = parent_request

        self.generated_tokens = 0
        self.prefill_finished_at = None
        self.generation_finished_at = None

        self.SLO = slo
        self.elapsed_time = 0
        self.urgency = 0

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
        pass

    def update_urgency(self, hardware_name: str):
        empirical_time = calculate_empirical_time(hardware_name, self.step)
        self.urgency = -((self.SLO - self.elapsed_time) - empirical_time)

    def __str__(self):
        return f"Request {self.req_id} for model {self.model} with input length {self.input_length} and output length {self.output_length} arrived at {self.arrive_at}"

    def __repr__(self) -> str:
        return self.__str__()

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
    current_stage: int
    total_stages: int
    total_time: float
    gen_requests_config: List[dict]
    current_requests: Optional[GenerationRequest] = None
    stage_lst: List[str] = None
    request_counter: int = 0
    
    def __init__(self, req_id: str, gen_requests_config: List[dict], slo=52.95, hardware_lst: List[str] = ["nvidia_A100"]):
        self.slo = slo
        self.req_id = req_id
        self.current_stage = 0
        self.stage_lst = []
        self.gen_requests_config = gen_requests_config
        self.total_time = 0.0
        self.current_requests = []
        self.request_counter = 0
        self._initialize_stages()
        self.total_stages = len(self.stage_lst)
        self.standard_workflow = STANDARD_WORKFLOW
        self.hardware_lst = hardware_lst
    
    def _initialize_stages(self):
        """Initialize the stages and maintain the order"""
        consecutive_steps = {"generate_candidate_llama-agent", "evaluate"}
        seen_steps = set()
        for stage_config in self.gen_requests_config:
            step = stage_config["step"]
            if step in consecutive_steps:
                if step not in seen_steps:
                    self.stage_lst.append(step)
                    seen_steps.add(step)
            else:
                self.stage_lst.append(step)
        # for stage_config in self.gen_requests_config:
        #     step = stage_config["step"]
        #     self.stage_lst.append(step)
    
    def create_current_stage_requests(self, model, arrive_at: float) -> List[Optional[GenerationRequest]]:
        """创建下一阶段的请求"""
        # Set a flag to distinguish between the first generate_candidate_llama-agent and the second one
        candidate_count = 0
        current_step = self.stage_lst[self.current_stage]
        if current_step in self.standard_workflow:
            self.standard_workflow.remove(current_step)
        if current_step == "unit_tester" and "revise" in self.standard_workflow:
            self.standard_workflow = list(filter(lambda x: x != "revise", self.standard_workflow))
        if current_step == "generate_candidate_llama-agent" or current_step == "evaluate":
            for stage_config in self.gen_requests_config:
                if stage_config["step"] == current_step:
                    if current_step == "generate_candidate_llama-agent" and candidate_count < 4:
                        next_request = GenerationRequest(
                            req_id=f"{self.req_id}_req_{self.request_counter}",
                            model=model,
                            step=current_step+'1',
                            slo=(self.slo - self.total_time) * (calculate_avg_empirical_time(self.hardware_lst, current_step) / (calculate_avg_empirical_time(self.hardware_lst, current_step) + sum([calculate_avg_empirical_time(self.hardware_lst, s) for s in self.standard_workflow]))),
                            input_length=stage_config["input_length"],
                            output_length=stage_config["output_length"],
                            arrive_at=arrive_at,
                            parent_request=self
                        )
                        candidate_count += 1
                    else:
                        next_request = GenerationRequest(
                            req_id=f"{self.req_id}_req_{self.request_counter}",
                            model=model,
                            step=current_step,
                            slo=(self.slo - self.total_time) * (calculate_avg_empirical_time(self.hardware_lst, current_step) / (calculate_avg_empirical_time(self.hardware_lst, current_step) + sum([calculate_avg_empirical_time(self.hardware_lst, s) for s in self.standard_workflow]))),
                            input_length=stage_config["input_length"],
                            output_length=stage_config["output_length"],
                            arrive_at=arrive_at,
                            parent_request=self
                        )
                    self.current_requests.append(next_request)
                    self.request_counter += 1
        else:
            next_request = GenerationRequest(
                req_id=f"{self.req_id}_req_{self.request_counter}",
                model=model,
                step=current_step,
                slo=(self.slo - self.total_time) * (calculate_avg_empirical_time(self.hardware_lst, current_step) / (calculate_avg_empirical_time(self.hardware_lst, current_step) + sum([calculate_avg_empirical_time(self.hardware_lst, s) for s in self.standard_workflow]))),
                input_length=self.gen_requests_config[self.request_counter]["input_length"],
                output_length=self.gen_requests_config[self.request_counter]["output_length"],
                arrive_at=arrive_at,
                parent_request=self
            )
            self.current_requests.append(next_request)
            self.request_counter += 1
        return self.current_requests
        # next_request = GenerationRequest(
        #     req_id=f"{self.req_id}_req_{self.request_counter}",
        #     model=model,
        #     step=current_step,
        #     slo=0.0,
        #     input_length=self.gen_requests_config[self.request_counter]["input_length"],
        #     output_length=self.gen_requests_config[self.request_counter]["output_length"],
        #     arrive_at=arrive_at,
        #     parent_request=self
        # )
        # self.current_requests.append(next_request)
        # self.request_counter += 1
        # return self.current_requests
    
    def update_stage(self, request, current_time: float):
        """更新阶段和总时间"""
        self.current_requests.remove(request)
        if self.current_requests == []:
            self.total_time += (current_time - request.arrive_at)
            self.current_stage += 1
