from typing import Callable, Tuple, Optional, List

# Numpy
import numpy as np
from dataclasses import replace

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class ChannelSelector(Operation):
    def __init__(self, channels: List[int], two_dims: bool = False):
        self.two_dims = two_dims
        self.channels = channels

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        two_dims = self.two_dims
        channels = np.array(self.channels, dtype=np.int64)

        def select_channels(images, *args):
            return images[:, channels] if two_dims else images[:, channels, ::]

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
