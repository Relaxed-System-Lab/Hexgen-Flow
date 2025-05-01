from typing import List
from simulator.core.request import GenerationRequest

class GlobalWaitlist:
    _instance = None

    def __init__(self):
        self.waitlist: List[GenerationRequest] = []

    @staticmethod
    def get_instance():
        if GlobalWaitlist._instance is None:
            GlobalWaitlist._instance = GlobalWaitlist()
        return GlobalWaitlist._instance

    def add_request(self, request: GenerationRequest):
        self.waitlist.append(request)

    def remove_request(self, request: GenerationRequest):
        self.waitlist.remove(request)

    def update_elapsed_time(self, current_time: float, hardware_name: str):
        for request in self.waitlist:
            request.elapsed_time = current_time - request.arrive_at
            request.update_urgency(hardware_name)

