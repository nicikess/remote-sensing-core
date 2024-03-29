import os
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from torch import nn

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class EsaWorldCoverModality(Modality):
    NAME = "esa_worldcover"

    def __init__(
        self,
        esa_world_cover_index_path: Union[str, Path],
        multiclass_label_threshold: float = 0.05,
        multiclass_label_top_k: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.esa_world_cover_index = pd.read_csv(esa_world_cover_index_path)

        label_vector = self.esa_world_cover_index.loc[[0]]
        label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
        self.number_of_classes = len(label_vector.columns)
        self.multiclass_label_threshold = multiclass_label_threshold
        self.multiclass_label_top_k = multiclass_label_top_k

    def _load(self, patch_id, *args, **kwargs):
        path_image_world_cover = (
            os.path.join(self.data_root_path, patch_id) + "_esaworldcover.npy"
        )
        world_cover_img = np.load(path_image_world_cover)
        return world_cover_img

    def get_world_cover_multiclass_label(self, patch_id):
        esa_row = self.esa_world_cover_index.loc[
            self.esa_world_cover_index["patch_id"] == patch_id
        ]
        assert len(esa_row) == 1
        numeric_label = esa_row.drop(["filename", "patch_id"], axis=1)
        # Set values to smaller than the threshold to 0
        label_vector = np.where(
            numeric_label <= self.multiclass_label_threshold, 0, numeric_label
        )
        numeric_label = np.squeeze(numeric_label.values)
        label_vector = np.squeeze(label_vector)
        # Get indexes of largest values
        max_indices = np.argpartition(label_vector, -self.multiclass_label_top_k)[
            -self.multiclass_label_top_k :
        ]
        # Create label encoding and set to one if value is not 0
        one_hot_label = np.zeros(self.number_of_classes)
        for i in range(len(max_indices)):
            if label_vector[max_indices[i]] > 0:
                one_hot_label[max_indices[i]] = 1
        if self.numpy_dtype:
            one_hot_label = one_hot_label.astype(self.numpy_dtype)
            numeric_label = numeric_label.astype(self.numpy_dtype)
        return one_hot_label, numeric_label


class EsaWorldCoverTransform(nn.Module):
    def __init__(
        self,
        divisor: Optional[Tuple[int]] = (10, 1),
        convert_to_label: Optional[bool] = False,
        transform: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.divisor = divisor
        self.transform = transform
        self.convert_to_label = convert_to_label

    def forward(self, x, *args, **kwargs):
        if self.divisor:
            x = (x / self.divisor[0]) - self.divisor[1]
        if self.convert_to_label:
            x = np.squeeze(x, axis=0)
            x = x.astype("int64")
        if self.transform:
            x = self.transform(x)
        return x
