from typing import Callable, Tuple, Optional

# Numpy
import numpy as np

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

class EsaWorldCoverTransform(Operation):

    def generate_code(self, divisor_and_subtractor: Tuple[int, int] = (10, 1)) -> Callable:
        # get local variables to use in return function
        divisor = divisor_and_subtractor[0]
        subtractor = divisor_and_subtractor[1]

        def esa_transform(images, *args):
            return (images / divisor) - subtractor

        return esa_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
