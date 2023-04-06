import os
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import rasterio
from torch import nn

# PyTorch
from torch.utils.data import Dataset

from remote_sensing_core.constants import (
    Bands,
    S1_IMG_KEY,
    S2_IMG_KEY,
    WORLD_COVER_IMG_KEY,
    ALTITUDE_IMG_KEY,
    STACKED_IMAGE_KEY,
    MULTICLASS_LABEL_KEY,
    NUMPY_DTYPE,
)


class BenGeS(Dataset):
    def __init__(
        self,
        root_dir_s1: Union[str, Path],
        root_dir_s2: Union[str, Path],
        root_dir_world_cover: Union[str, Path],
        root_dir_glo_30_dem: Union[str, Path],
        era5_data_path: Union[str, Path],
        esa_world_cover_index_path: Union[str, Path],
        sentinel_1_2_metadata_path: Union[str, Path],
        s2_bands: Union[str, Bands],
        transform: Optional[nn.Module] = None,
        multiclass_label_threshold: float = 0.3,
        multiclass_label_top_k: int = 3,
        normalization_value: float = 10000.0,
        stacked_modalities: Optional[List] = None,
    ):
        self.root_dir_s1 = root_dir_s1
        self.root_dir_s2 = root_dir_s2
        self.root_dir_world_cover = root_dir_world_cover
        self.root_dir_glo_30_dem = root_dir_glo_30_dem
        # Read in csv for climate data
        self.era5_data = pd.read_csv(era5_data_path)

        # Read in csv files for indexing
        self.esa_world_cover_index = pd.read_csv(esa_world_cover_index_path)
        self.sentinel_1_2_metadata = pd.read_csv(sentinel_1_2_metadata_path)

        # Get number of classes
        label_vector = self.esa_world_cover_index.loc[[0]]
        label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
        self.number_of_classes = len(label_vector.columns)

        self.s2_bands = Bands(s2_bands)
        self.transform = transform
        self.multiclass_label_threshold = multiclass_label_threshold
        self.multiclass_label_top_k = multiclass_label_top_k
        self.normalization_value = normalization_value

        # Parameters which control output format
        self.stacked_modalities = stacked_modalities

    def __len__(self):
        return len(self.esa_world_cover_index)

    def __getitem__(self, idx):
        # Load modalities
        patch_id = self.esa_world_cover_index.loc[:, "patch_id"][idx]

        # Load, convert and normalize sentinel 1 and 2 images
        img_s1, img_s2 = (
            self._transform_numpy_images(img)
            for img in (
                self._load_sentinel_1_image(patch_id),
                self._load_sentinel_2_image(patch_id),
            )
        )

        # World cover
        img_world_cover = (
            np.load(
                os.path.join(self.root_dir_world_cover, patch_id) + "_esaworldcover.npy"
            )
            / 10.0
            - 1
        )

        # Altitude data from glo_dem_30 model
        img_altitude = self._load_altitude_image(patch_id)

        # TODO Climate data

        # Define output tensor
        output_tensor = {
            S1_IMG_KEY: img_s1,
            S2_IMG_KEY: img_s2,
            WORLD_COVER_IMG_KEY: img_world_cover,
            ALTITUDE_IMG_KEY: img_altitude,
        }
        if self.stacked_modalities:
            output_tensor = {
                STACKED_IMAGE_KEY: np.concatenate(
                    [output_tensor[k] for k in self.stacked_modalities]
                )
            }
        # Get multiclass label for patch
        output_tensor[MULTICLASS_LABEL_KEY] = self._derive_multiclass_label(idx)
        return output_tensor

    def _load_sentinel_1_image(self, patch_id):
        file_name_s1 = self.sentinel_1_2_metadata.loc[
            self.sentinel_1_2_metadata["patch_id"] == patch_id, "patch_id_s1"
        ].values[0]
        path_image_s1 = os.path.join(self.root_dir_s1, file_name_s1) + "_all_bands.npy"
        return np.load(path_image_s1)

    def _load_sentinel_2_image(self, patch_id):
        path_image_s2 = os.path.join(self.root_dir_s2, patch_id) + "_all_bands.npy"
        img_s2 = np.load(path_image_s2)
        if self.s2_bands == Bands.RGB:
            img_s2 = img_s2[[3, 2, 1], :, :]
            assert img_s2.shape == (3, 120, 120), print("False shape:", img_s2.shape)
        if self.s2_bands == Bands.INFRARED:
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if self.s2_bands != Bands.ALL:
            raise NotImplementedError(f"S2 image bands {self.s2_bands} not implemented")
        return img_s2

    def _load_altitude_image(self, patch_id):
        with rasterio.open(
            os.path.join(self.root_dir_glo_30_dem, patch_id + "_dem.tif")
        ) as image_file:
            img_altitude = image_file.read()
        return img_altitude

    def _transform_numpy_images(self, img: np.array):
        img = img.astype(NUMPY_DTYPE)
        img = img / self.normalization_value
        if self.transform:
            img = self.transform(img)
        return img

    def _derive_multiclass_label(self, idx):
        label_vector = self.esa_world_cover_index.loc[[idx]]
        label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
        # Set values to smaller than the threshold to 0
        label_vector = np.where(
            label_vector <= self.multiclass_label_threshold, 0, label_vector
        )
        label_vector = np.squeeze(label_vector)
        # Get indexes of largest values
        max_indices = np.argpartition(label_vector, -self.multiclass_label_top_k)[
            -self.multiclass_label_top_k :
        ]
        # Create label encoding and set to one if value is not 0
        label = np.zeros(self.number_of_classes)
        for i in range(len(max_indices)):
            if label_vector[max_indices[i]] > 0:
                label[max_indices[i]] = 1
        return label.astype(NUMPY_DTYPE)
