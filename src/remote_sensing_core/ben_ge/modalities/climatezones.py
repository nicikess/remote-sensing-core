import pandas as pd

# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class ClimateZoneModality(Modality):
    NAME = "climate_zone"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.data_root_path
        self.climate_zone_data = pd.read_csv(self.data_root_path)

    def _load(self, patch_id, *args, **kwargs):
        climate_zone = self.climate_zone_data.loc[
            self.climate_zone_data["patch_id"] == patch_id, "climatezone"
        ].values[0]
        return climate_zone
