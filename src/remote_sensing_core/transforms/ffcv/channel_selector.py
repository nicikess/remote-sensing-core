from typing import Callable, Tuple, Optional, Union

# Numpy
import numpy as np

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
            if bands == Bands.RGB:
                images = images[:, [3, 2, 1], :, :]
                images_no_batch_size = np.squeeze(images, axis=0)
                assert images_no_batch_size.shape == (3, 120, 120), print("False shape:", images.shape)
            if bands == Bands.INFRARED:
                images = images[:, [7, 3, 2, 1], :, :]
                images_no_batch_size = np.squeeze(images, axis=0)
                assert images_no_batch_size.shape == (4, 120, 120), print("False shape:", images.shape)
            if bands not in Bands:
                raise NotImplementedError(f"S2 image bands {self.s2_bands} not implemented")
            return images

        return select_channels


    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
