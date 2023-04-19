from typing import Callable, Tuple, Optional

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class MinMaxScaler(Operation):
    def __init__(
        self,
        minimum_value: float,
        maximum_value: float,
        interval_min: float = -1.0,
        interval_max: float = 1.0,
    ):
        # Store Scaler values
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.interval_min = interval_min
        self.interval_max = interval_max

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        minimum = self.minimum
        maximum = self.maximum
        interval_minimum = self.minimum_value
        interval_maximum = self.maximum_value

        def scale_images(images, *args):
            return ((images - minimum) / (maximum - minimum)) * (
                interval_maximum - interval_minimum
            ) + interval_minimum

        scale_images.is_parallel = True
        return scale_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
