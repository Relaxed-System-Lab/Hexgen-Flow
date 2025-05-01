from ._base import BaseGlobalToLocalPolicy
from simulator.core.request import GenerationRequest
from simulator.core.engine import LLMEngine
from typing import Dict, List
import itertools


class EvenGTLPolicy(BaseGlobalToLocalPolicy):
    def __init__(self):
        super().__init__()
        self.engines = None
        self.engine_cycle = None

    def prepare(self, engines: Dict[str, List[LLMEngine]]):
        self.engines = engines
        self.engine_cycle = {model: itertools.cycle(engine_list) for model, engine_list in engines.items()}

    def assign_requests(self, request: GenerationRequest):
        selected_engine = next(self.engine_cycle[request.model])
        selected_engine.add_request(request)
