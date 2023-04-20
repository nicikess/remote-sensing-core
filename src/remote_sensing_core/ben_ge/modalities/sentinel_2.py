import os
from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np
from torch import nn

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality
from remote_sensing_core.constants import Bands


class Sentinel2Modality(Modality):
    NAME = "sentinel_2"

    def __init__(
        self, s2_bands: Union[str, Bands], *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.s2_bands = Bands(s2_bands)

    def load_sample(self, patch_id, *args, **kwargs):
        path_image_s2 = self.data_root_path.joinpath(
            patch_id, patch_id + "_all_bands.npy"
        )
        img_s2 = np.load(path_image_s2)
        if self.s2_bands == Bands.RGB:
            img_s2 = img_s2[[3, 2, 1], :, :]
            assert img_s2.shape == (3, 120, 120), print("False shape:", img_s2.shape)
        if self.s2_bands == Bands.INFRARED:
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if self.s2_bands not in Bands:
            raise NotImplementedError(f"S2 image bands {self.s2_bands} not implemented")
        return self.transform_sample(img_s2)


class Sentinel2Transform(nn.Module):
    def __init__(
        self,
        clip_values: Optional[Tuple[float]] = None,
        normalization_value: Optional[float] = None,
        transform: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.normalization_value = normalization_value
        self.clip_values = clip_values
        self.transform = transform

    def forward(self, x, *args, **kwargs):
        if self.clip_values:
            x = x.clip(*self.clip_values)
        if self.normalization_value:
            x /= self.normalization_value
        if self.transform:
            x = self.transform(x)
        return x
