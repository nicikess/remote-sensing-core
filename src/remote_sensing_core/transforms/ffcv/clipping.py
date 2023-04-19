from typing import Callable, Tuple, Optional

# Numpy
import numpy as np

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class Clipping(Operation):
    def __init__(self, clip_values: Tuple[int, int]):
        self.minimum, self.maximum = clip_values

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        minimum = self.minimum
        maximum = self.maximum

        def clip_images(images, *args):
            return np.clip(images, a_min=minimum, a_max=maximum)

        clip_images.is_parallel = True
        return clip_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
