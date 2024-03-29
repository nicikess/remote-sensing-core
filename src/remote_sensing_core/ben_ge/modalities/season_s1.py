# Remote sensing core
from remote_sensing_core.ben_ge.modalities.modality import Modality


class SeasonS1Modality(Modality):
    NAME = "season_s1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self, patch_id, *args, **kwargs):
        sentinel_1_2_metadata = kwargs["sentinel_1_2_metadata"]
        season_s1 = sentinel_1_2_metadata.loc[
            sentinel_1_2_metadata["patch_id"] == patch_id, "season_s1"
        ].values[0]
        return season_s1
