from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd
from torch import nn

# PyTorch
from torch.utils.data import Dataset

from remote_sensing_core.ben_ge.modalities.climatezones import ClimateZoneModality
from remote_sensing_core.ben_ge.modalities.sentinel_1 import Sentinel1Modality
from remote_sensing_core.ben_ge.modalities.season_s1 import SeasonS1Modality
from remote_sensing_core.ben_ge.modalities.season_s2 import SeasonS2Modality
from remote_sensing_core.ben_ge.modalities.sentinel_2 import Sentinel2Modality
from remote_sensing_core.ben_ge.modalities.esa_worldcover import EsaWorldCoverModality
from remote_sensing_core.ben_ge.modalities.glo_30_dem import Glo30DemModality
from remote_sensing_core.ben_ge.modalities.era5 import Era5Modality
from remote_sensing_core.ben_ge.modalities.modality import Modality
from remote_sensing_core.constants import (
    MULTICLASS_ONE_HOT_LABEL_KEY,
    MULTICLASS_NUMERIC_LABEL_KEY,
    ELEVATION_DIFFERENCE_LABEL_KEY,
)


class BenGe(Dataset):
    def __init__(
        self,
        data_index_path: Union[str, Path],
        sentinel_1_2_metadata_path: Union[str, Path],
        sentinel_1_modality: Optional[Sentinel1Modality] = None,
        season_s1_modality: Optional[SeasonS2Modality] = None,
        season_s2_modality: Optional[SeasonS2Modality] = None,
        sentinel_2_modality: Optional[Sentinel2Modality] = None,
        esa_world_cover_modality: Optional[EsaWorldCoverModality] = None,
        glo_30_dem_modality: Optional[Glo30DemModality] = None,
        era_5_modality: Optional[Era5Modality] = None,
        climate_zone_modality: Optional[ClimateZoneModality] = None,
        transform: Optional[nn.Module] = None,
        output_as_tuple: bool = False,
    ):
        # Read in csv files for indexing
        self.data_index = pd.read_csv(data_index_path)
        self.sentinel_1_2_metadata = pd.read_csv(sentinel_1_2_metadata_path)
        self.transform = transform
        self.output_as_tuple = output_as_tuple

        # Register modalities
        self.modalities_dict: Dict[str, Modality] = {}
        # Sentinel 1
        if sentinel_1_modality:
            self.modalities_dict[Sentinel1Modality.NAME] = sentinel_1_modality
        # S1 Season Data
        if season_s1_modality:
            self.modalities_dict[SeasonS1Modality.NAME] = season_s1_modality
        # Sentinel 2
        if sentinel_2_modality:
            self.modalities_dict[Sentinel2Modality.NAME] = sentinel_2_modality
        # S2 Season Data
        if season_s2_modality:
            self.modalities_dict[SeasonS2Modality.NAME] = season_s2_modality
        # WorldCover
        if esa_world_cover_modality:
            self.modalities_dict[EsaWorldCoverModality.NAME] = esa_world_cover_modality
        # Altitude Model
        if glo_30_dem_modality:
            self.modalities_dict[Glo30DemModality.NAME] = glo_30_dem_modality
        # Climate data
        if era_5_modality:
            self.modalities_dict[Era5Modality.NAME] = era_5_modality
        # Climate zone
        if climate_zone_modality:
            self.modalities_dict[ClimateZoneModality.NAME] = climate_zone_modality

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # Load modalities
        patch_id = self.data_index.loc[:, "patch_id"][idx]

        # Define output tensor
        output_tensor = {
            modality_name: self.transform(
                modality_class.load_sample(
                    patch_id=patch_id, sentinel_1_2_metadata=self.sentinel_1_2_metadata
                )
            )
            if self.transform
            else modality_class.load_sample(
                patch_id=patch_id, sentinel_1_2_metadata=self.sentinel_1_2_metadata
            )
            for modality_name, modality_class in self.modalities_dict.items()
        }

        # Get esa world cover labels
        if EsaWorldCoverModality.NAME in self.modalities_dict.keys():
            ewc_modality = self.modalities_dict[EsaWorldCoverModality.NAME]
            assert isinstance(ewc_modality, EsaWorldCoverModality)
            (
                one_hot_label,
                numeric_label,
            ) = ewc_modality.get_world_cover_multiclass_label(patch_id=patch_id)
            output_tensor[MULTICLASS_ONE_HOT_LABEL_KEY] = one_hot_label
            output_tensor[MULTICLASS_NUMERIC_LABEL_KEY] = numeric_label

        # Get Elevation labels
        if Glo30DemModality.NAME in self.modalities_dict.keys():
            elevation_modality = self.modalities_dict[Glo30DemModality.NAME]
            assert isinstance(elevation_modality, Glo30DemModality)
            output_tensor[
                ELEVATION_DIFFERENCE_LABEL_KEY
            ] = elevation_modality.get_elevation_difference_in_patch(patch_id=patch_id)

        # Return as tuple if desired
        if self.output_as_tuple:
            mapping = []
            output_tensor_tuple = tuple()
            for k, v in dict(
                sorted(output_tensor.items(), key=lambda x: x[0].lower())
            ).items():
                mapping.append(k)
                output_tensor_tuple += (v,)
            output_tensor_tuple += (mapping,)
            return output_tensor_tuple
        return output_tensor

    def __str__(self):
        return list(self.modalities_dict.keys())
