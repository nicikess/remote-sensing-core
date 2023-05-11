from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import CLIMATE_ZONE_INDEX

import numpy as np

class ClimateZonesTransform(Operation):
    def __init__(self):
        # values taken from ben-ge_era-5.csv from ben-ge-100
        self.max_value = 29
        self.min_value = 0

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        max_value = self.max_value
        min_value = self.min_value

        def climate_zone_transform(climate_zone_batch, *args):
            climate_zone_batch_normalized = (climate_zone_batch - min_value) / (max_value - min_value)
            return climate_zone_batch_normalized

        return climate_zone_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
