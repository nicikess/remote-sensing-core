from typing import Callable, Tuple, Optional, Union

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

    def __init__(self, s2_bands: Union[str, Bands]):
        self.band_names = s2_bands

    def generate_code(self) -> Callable:
        # get local variables to use in return function
        bands = self.band_names

        def select_channels(images, *args):
            print(np.shape(images))
            if bands == Bands.RGB:
                rgb_channels = [3, 2, 1]
                rgb_channels = np.array(rgb_channels, dtype=np.int64)
                images = images[:, rgb_channels, :, :]
            if bands == Bands.INFRARED:
                infrared_channels = [7, 3, 2, 1]
                infrared_channels = np.array(infrared_channels, dtype=np.int64)
                images = images[:, infrared_channels, :, :]
            return images

        return select_channels

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:

        c, h, w = previous_state.shape

        shape = (12, h, w)
        if self.band_names == Bands.RGB:
            shape = (3, h, w)
        if self.band_names == Bands.INFRARED:
            shape = (4, h, w)

        # Update state shape
        new_state = replace(previous_state, shape=shape)

        # Allocate memory for new image
        mem_allocation = AllocationQuery(shape, dtype=previous_state.dtype)

        return new_state, mem_allocation
