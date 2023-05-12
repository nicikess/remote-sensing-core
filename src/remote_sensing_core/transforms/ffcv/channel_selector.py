from typing import Callable, Tuple, Optional, List

# Numpy
import numpy as np
from dataclasses import replace

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

# Bands
from remote_sensing_core.constants import Bands


class ChannelSelector(Operation):
    def __init__(self, channels: List[int]):
        self.channels = channels

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        channels = np.array(self.channels, dtype=np.int64)

        def select_channels(images, *args):
            print(images.shape)
            if len(images.shape) > 3:
                return images[:, channels, ::]
            else:
                return images[:, channels]

        return select_channels

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        # Get new shape
        shape = (len(self.channels), *previous_state.shape[1:])

        # Update state shape
        new_state = replace(previous_state, shape=shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(shape, dtype=previous_state.dtype)

        return new_state, mem_allocation
