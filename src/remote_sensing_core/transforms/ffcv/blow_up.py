from dataclasses import replace
from typing import Callable, Tuple, Optional, List

# Numpy
import numpy as np

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State
from ffcv.pipeline import Compiler


class BlowUp(Operation):
    def __init__(self, new_shape: List[int]):
        assert len(new_shape) == 3, "New shape needs to resemble an image [C x H x W]"
        self.new_shape = new_shape

    def generate_code(self) -> Callable:
        parallel_batch_range = Compiler.get_iterator()
        parallel_channel_range = Compiler.get_iterator()

        # get local variables to use in return function
        n_c, n_h, n_w = self.new_shape

        def blow_up_images(values, *args):
            b, c = values.shape
            assert c == n_c, "Channel dimension has to match"

            new_values = np.full(shape=(b, c, n_h, n_w), fill_value=-1.0,)
            # create blown up values
            for b_idx in parallel_batch_range(b):
                sample = values[b_idx]
                for c_idx in parallel_channel_range(c):
                    new_values[b_idx][c_idx] = np.full(
                        shape=(n_h, n_w), fill_value=sample[c_idx], dtype=values.dtype
                    )
            return new_values

        # blow_up_images.is_parallel = True
        return blow_up_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        # Update state shape
        new_state = replace(previous_state, shape=self.new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(self.new_shape, dtype=previous_state.dtype)
        return new_state, mem_allocation
