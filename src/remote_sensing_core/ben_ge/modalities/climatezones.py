# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class ClimateZoneModality(Modality):
    NAME = "climate_zone"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_sample(self, patch_id, *args, **kwargs):
        sentinel_1_2_metadata = kwargs["sentinel_1_2_metadata"]
        climate_zone = sentinel_1_2_metadata.loc[
            sentinel_1_2_metadata["patch_id"] == patch_id, "climatezone"
        ].values[0]
        return climate_zone
