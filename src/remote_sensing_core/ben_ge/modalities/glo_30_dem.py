import os
from typing import Union
from pathlib import Path

import rasterio as rio
import pandas as pd

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class Glo30DemModality(Modality):
    NAME = "glo_30_dem"

    def __init__(self, data_root_path: Union[str, Path], *args, **kwargs):
        super().__init__(data_root_path=data_root_path, *args, **kwargs)

    def load_sample(self, patch_id, *args, **kwargs):
        with rio.open(
            os.path.join(self.data_root_path, patch_id + "_dem.tif")
        ) as image_file:
            img_altitude = image_file.read()
        return self.transform_sample(img_altitude)
