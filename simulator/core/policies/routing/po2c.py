import itertools
import random
from typing import Dict, List

from ._base import BaseGlobalToLocalPolicy
from simulator.core.request import GenerationRequest
from simulator.core.request import calculate_empirical_time
from simulator.core.engine import LLMEngine


class Po2CGTLPolicy(BaseGlobalToLocalPolicy):
    def __init__(self):
        super().__init__()
        self.engines = None
        self._calculate_empirical_time = calculate_empirical_time

    def prepare(self, engines: Dict[str, List[LLMEngine]]):
        self.engines = engines

    def _load(self, engine: LLMEngine) -> int:
        workload = 0.0
        for request in itertools.chain(engine.waiting, engine.running):
            workload += self._calculate_empirical_time(engine.hardware_name, request.step)
        return workload

    def set_empirical_time_fn(self, fn):
        self._calculate_empirical_time = fn

    def assign_requests(self, request: GenerationRequest):
        candidates = self.engines[request.model]
        if len(candidates) == 1:
            candidates[0].add_request(request)
            return
        left, right = random.sample(candidates, 2)
        chosen = left if self._load(left) <= self._load(right) else right
        chosen.add_request(request)
