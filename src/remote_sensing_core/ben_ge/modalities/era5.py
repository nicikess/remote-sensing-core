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

    def load_sample(self, patch_id, *args, **kwargs):
        raise NotImplementedError