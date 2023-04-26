import os
from typing import Union
from pathlib import Path

import rasterio as rio
import pandas as pd

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class Glo30DemModality(Modality):
    NAME = "glo_30_dem"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self, patch_id, *args, **kwargs):
        with rio.open(
            os.path.join(self.data_root_path, patch_id + "_dem.tif")
        ) as image_file:
            img_altitude = image_file.read()
        return img_altitude

    def get_elevation_difference_in_patch(self, patch_id):
        with rio.open(
            os.path.join(self.data_root_path, patch_id + "_dem.tif")
        ) as image_file:
            img_altitude = image_file.read()[0]
        return img_altitude.max() - img_altitude.min()
