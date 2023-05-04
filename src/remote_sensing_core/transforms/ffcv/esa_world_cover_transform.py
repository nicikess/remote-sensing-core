from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class EsaWorldCoverTransform(Operation):
    def __init__(self, divisor: float, subtractor: float):
        self.subtractor = subtractor
        self.divisor = divisor

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        divisor = self.divisor
        subtractor = self.subtractor

        def esa_transform(images, *args):
            return (images / divisor) - subtractor

        return esa_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
