from typing import Callable, Tuple, Optional, Union

# Numpy
import numpy as np
from dataclasses import replace

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class ExpandDimension(Operation):
    def __init__(self, axis: int = 1):
        assert axis > 0
        self.axis = axis

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        axis = self.axis

        def expand_dimension(images, *args):
            images = np.expand_dims(images, axis=axis)
            return images

        return expand_dimension

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = [*previous_state.shape]
        axis_index = self.axis-1
        new_shape = tuple([*new_shape[:axis_index], 1, *new_shape[axis_index:]])

        # Update state shape
        new_state = replace(previous_state, shape=new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(new_shape, dtype=previous_state.dtype)

        return new_state, mem_allocation