from typing import Union
from pathlib import Path

import pandas as pd

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class Era5Modality(Modality):
    NAME = "era_5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.era5_data = pd.read_csv(self.data_root_path)

    def _load(self, patch_id, *args, **kwargs):
        weather_data = self.era5_data.loc[
            self.era5_data["patch_id"] == patch_id
        ].values[0]
        return self.transform_sample(weather_data[2:])  # exclude patch ids
