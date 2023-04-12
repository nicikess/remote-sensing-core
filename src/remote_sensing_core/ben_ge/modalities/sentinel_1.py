import os
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class Sentinel1Modality(Modality):
    NAME = "sentinel_1"

    def __init__(
        self,
        sentinel_1_2_metadata_path: Union[str, Path],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sentinel_1_2_metadata = pd.read_csv(sentinel_1_2_metadata_path)

    def load_sample(self, patch_id, *args, **kwargs):
        file_name_s1 = self.sentinel_1_2_metadata.loc[
            self.sentinel_1_2_metadata["patch_id"] == patch_id, "patch_id_s1"
        ].values[0]
        path_image_s1 = self.data_root_path.join(file_name_s1) + "_all_bands.npy"
        return self.transform_sample(np.load(path_image_s1))
