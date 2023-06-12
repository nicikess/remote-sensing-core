from typing import Callable, List, Optional, Tuple

# Nmpy
import numpy as np

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler


class ChannelWiseMinMaxScaler(Operation):
    def __init__(
        self,
        minimum_value: List[float],
        maximum_value: List[float],
        interval_min: List[float] = None,
        interval_max: List[float] = None,
        two_dims: bool = False,
    ):
        self.two_dims = two_dims
        assert len(minimum_value) == len(maximum_value)
        # Store Scaler values
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.interval_min = (
            interval_min if interval_min else [-1 for _ in range(len(maximum_value))]
        )
        self.interval_max = (
            interval_max if interval_min else [1 for _ in range(len(maximum_value))]
        )
        assert len(self.interval_min) == len(self.interval_max)
        assert len(self.interval_max) == len(minimum_value)

    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        # get local variables to use in return function
        two_dims = self.two_dims
        n_channels = len(self.maximum_value)
        minimum = self.minimum_value
        maximum = self.maximum_value
        interval_minimum = self.interval_min
        interval_maximum = self.interval_max

        def scale_images(images, *args):
            results = []
            for i in parallel_range(n_channels):
                current_minimum = minimum[i]
                current_maximum = maximum[i]
                current_interval_minimum = interval_minimum[i]
                current_interval_maximum = interval_maximum[i]

                current_channels = images[:, i] if two_dims else images[:, i, ::]
                results.append(
                    (
                        (current_channels - current_minimum)
                        / (current_maximum - current_minimum)
                    )
                    * (current_interval_maximum - current_interval_minimum)
                    + current_interval_minimum
                )
            return np.stack(results, axis=1,)

        scale_images.is_parallel = True
        return scale_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
