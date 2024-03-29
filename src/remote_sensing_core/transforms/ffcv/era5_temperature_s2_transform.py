from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import TEMPERATURE_S2_INDEX


class Era5TemperatureS2Transform(Operation):
    def __init__(
        self, batch_size: int,
    ):
        # values taken from ben-ge_era-5.csv from ben-ge-100
        self.max_value = 304.03
        self.min_value = 261.625
        self.batch_size = batch_size

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        max_value = self.max_value
        min_value = self.min_value
        batch_size = self.batch_size

        def era5_transform(row, *args):
            # access temperature s2 at index 3
            temperature_s2 = row[:, TEMPERATURE_S2_INDEX]
            normalized_temperature = (temperature_s2 - min_value) / (
                max_value - min_value
            )
            normalized_temperature = normalized_temperature.reshape((batch_size, 1))
            return normalized_temperature

        return era5_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
