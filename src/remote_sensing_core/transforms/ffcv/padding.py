from dataclasses import replace
from typing import Callable, Tuple, Optional

# Numpy
import numpy as np

# FFCV
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class Padding(Operation):
    def __init__(self, padding: int, padding_value: float = 0.0):
        self.padding = padding
        self.pad_width = [[0, 0], [padding, padding], [padding, padding]]
        self.padding_value = padding_value

    def generate_code(self) -> Callable:
        image_range = Compiler.get_iterator()

        # get local variables to use in return function
        pad_width = self.pad_width
        padding_value = self.padding_value

        def pad_images(images, *args):
            # Expects images to be a batch of images of shape CxHxW
            batch_size, *_ = images.shape
            # Pad each image in batch in parallel
            for i in image_range(batch_size):
                images[i] = np.pad(
                    images[i],
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=padding_value,
                )

        pad_images.is_parallel = True
        return pad_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        c, h, w = previous_state.shape
        new_shape = (c, h + 2 * self.padding, w + 2 * self.padding)
        print("New shape", new_shape)

        # Update state shape
        new_state = replace(previous_state, shape=new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(new_shape, dtype=previous_state.dtype)
        return new_state, mem_allocation
