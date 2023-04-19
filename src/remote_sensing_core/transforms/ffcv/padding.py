from dataclasses import replace
from typing import Callable, Tuple, Optional

# Numpy
import numpy as np

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class Padding(Operation):
    def __init__(self, padding: int, padding_value: float = 0.0):
        self.padding = padding
        self.padding_value = padding_value

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        padding = self.padding
        padding_value = self.padding_value

        def pad_images(images, *args):
            # Expects images to be a batch of images of shape CxHxW
            b, c, h, w = images.shape

            # create new images
            new_images = np.full(
                shape=(b, c, int(h + 2 * padding), int(w + 2 * padding)),
                fill_value=padding_value,
                dtype=images.dtype,
            )
            new_images[:, :, padding:-padding, padding:-padding] = images
            return new_images

        pad_images.is_parallel = True
        return pad_images

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        c, h, w = previous_state.shape
        new_shape = (c, int(h + 2 * self.padding), int(w + 2 * self.padding))

        # Update state shape
        new_state = replace(previous_state, shape=new_shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(new_shape, dtype=previous_state.dtype)
        return new_state, mem_allocation
