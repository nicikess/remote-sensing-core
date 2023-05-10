from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import CLIMATE_ZONE_INDEX

class ClimateZonesTransform(Operation):
    def __init__(self):
        # values taken from ben-ge_era-5.csv from ben-ge-100
        self.max_value = 29
        self.min_value = 0

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        max_value = self.max_value
        min_value = self.min_value

        def climate_zone_transform(row, *args):
            # access temperature s2 at index 3
            climate_zone = row[CLIMATE_ZONE_INDEX]
            normalized_climate_zone = (climate_zone - min_value) / (max_value - min_value)
            grey_scaled_climate_zone = normalized_climate_zone * 255
            return grey_scaled_climate_zone

        return climate_zone_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
