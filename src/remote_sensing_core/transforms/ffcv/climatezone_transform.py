from typing import Callable, Tuple, Optional


# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import TEMPERATURE_S2_INDEX

import numpy as np
from dataclasses import replace

class Era5TemperatureS2Transform(Operation):
    def __init__(self):
        # values taken from ben-ge_era-5.csv from ben-ge-100
        self.number_of_climate_zones = 30

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        number_of_climate_zones = self.number_of_climate_zones

        def climate_zone_transform(batched_data, *args):
            return np.eye(number_of_climate_zones)[batched_data.flatten()]

        return climate_zone_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = self.number_of_climate_zones

        # Update state shape
        new_state = replace(previous_state, shape=new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(new_shape, dtype=previous_state.dtype)
        return new_state, mem_allocation
