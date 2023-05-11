from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import TEMPERATURE_S2_INDEX

class Era5TemperatureS2Transform(Operation):
    def __init__(self):
        # values taken from ben-ge_era-5.csv from ben-ge-100
        self.max_value = 304.03
        self.min_value = 261.625

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        max_value = self.max_value
        min_value = self.min_value

        def era5_transform(row, *args):
            normalized_temperature = (row - min_value) / (max_value - min_value)
            return normalized_temperature

        return era5_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
