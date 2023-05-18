from typing import Callable, Tuple, Optional, Union

# Numpy
import numpy as np
from dataclasses import replace

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class Remove1dChannel(Operation):
    def __init__(self, axis: int = 1):
        self.axis = axis

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        axis = self.axis

        def remove_1d_channel(images, *args):
            images = images[:, 0]
            return images

        return remove_1d_channel

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:

        c, h, w = previous_state.shape
        new_shape = (h, w)

        # Update state shape
        new_state = replace(previous_state, shape=new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(new_shape, dtype=previous_state.dtype)

        return new_state, mem_allocation