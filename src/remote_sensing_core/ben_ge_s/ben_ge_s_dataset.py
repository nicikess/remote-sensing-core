import os
from pathlib import Path
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

# PyTorch
from torch.utils.data import Dataset

from remote_sensing_core.constants import (
    Bands,
    S1_IMG_KEY,
    S2_IMG_KEY,
    WORLD_COVER_IMG_KEY,
    LABEL_KEY,
    NUMPY_DTYPE,
)


class BenGeS(Dataset):
    def __init__(
        self,
        root_dir_s1: Union[str, Path],
        root_dir_s2: Union[str, Path],
        root_dir_world_cover: Union[str, Path],
        era5_data_path: Union[str, Path],
        esa_world_cover_index_path: Union[str, Path],
        sentinel_1_2_metadata_path: Union[str, Path],
        s2_bands: Union[str, Bands],
        transform: nn.Module,
        normalization_value: float = 10000.0,
    ):
        self.root_dir_s1 = root_dir_s1
        self.root_dir_s2 = root_dir_s2
        self.root_dir_world_cover = root_dir_world_cover
        # Read in csv for climate data
        self.era5_data = pd.read_csv(era5_data_path)
        # TODO add altitude model root path

        # Read in csv files for indexing
        self.esa_world_cover_index = pd.read_csv(esa_world_cover_index_path)
        self.sentinel_1_2_metadata = pd.read_csv(sentinel_1_2_metadata_path)

        self.number_of_classes = ...  # TODO derive from random worldcover data point
        self.bands = s2_bands
        self.transform = transform
        self.normalization_value = normalization_value

    def __len__(self):
        return len(self.esa_world_cover_index)

    def __getitem__(self, idx):

        # Sentinel 2
        file_name_s2 = self.esa_world_cover_index.loc[:, "patch_id"][idx]
        path_image_s2 = os.path.join(self.root_dir_s2, file_name_s2) + "_all_bands.npy"
        img_s2 = np.load(path_image_s2)

        # Load other modalities

        # Sentinel 1
        file_name_s1 = self.sentinel_1_2_metadata.loc[
            self.sentinel_1_2_metadata["patch_id"] == file_name_s2, "patch_id_s1"
        ].values[0]
        path_image_s1 = os.path.join(self.root_dir_s1, file_name_s1) + "_all_bands.npy"
        img_s1 = np.load(path_image_s1)

        # World cover
        file_name_world_cover = self.esa_world_cover_index.loc[:, "patch_id"][idx]
        path_image_world_cover = (
            os.path.join(self.root_dir_world_cover, file_name_world_cover)
            + "_esaworldcover.npy"
        )
        img_world_cover = np.load(path_image_world_cover)

        # Encode label
        threshold = 0.3
        label_vector = self.esa_world_cover_index.loc[[idx]]
        label_vector = label_vector.drop(["filename", "patch_id"], axis=1)
        # Set values to smaller than the threshold to 0
        label_vector = np.where(label_vector <= threshold, 0, label_vector)
        label_vector = np.squeeze(label_vector)
        # Get indexes of largest values
        max_indices = np.argpartition(label_vector, -3)[-3:]
        # Create label encoding and set to one if value is not 0
        label = np.zeros(self.number_of_classes)
        for i in range(len(max_indices)):
            if label_vector[max_indices[i]] > 0:
                label[max_indices[i]] = 1
        label = label.astype(NUMPY_DTYPE)

        if self.bands == Bands.RGB:
            img_s2 = img_s2[[3, 2, 1], :, :]
            assert img_s2.shape == (3, 120, 120), print("False shape:", img_s2.shape)
        if self.bands == Bands.INFRARED:
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if self.bands == Bands.ALL:
            img_s2 = img_s2

        # change type of img
        img_s1 = img_s1.astype(NUMPY_DTYPE)
        img_s2 = img_s2.astype(NUMPY_DTYPE)
        img_world_cover = img_world_cover.astype(NUMPY_DTYPE)
        # Normalize img
        img_s1_normalized = img_s1 / self.normalization_value
        img_s2_normalized = img_s2 / self.normalization_value
        img_world_cover_normalized = img_world_cover / self.normalization_value

        if self.transform:
            img_s1_normalized = self.transform(img_s1_normalized)
            img_s2_normalized = self.transform(img_s2_normalized)
            img_world_cover_normalized = self.transform(img_world_cover_normalized)

        # Define output tensor
        output_tensor = {
            S1_IMG_KEY: img_s1_normalized,
            S2_IMG_KEY: img_s2_normalized,
            WORLD_COVER_IMG_KEY: img_world_cover_normalized,
            LABEL_KEY: label,
        }

        return output_tensor

    def display(self, idx):
        # TODO check whether this works
        sample = self[idx]
        imgdata = sample[S2_IMG_KEY]
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(
            1.3
            * (imgdata - np.min(imgdata, axis=(0, 1)))
            / (np.max(imgdata, axis=(0, 1)) - np.min(imgdata, axis=(0, 1)))
            + 0.1
        )
        ax.set_title("Sentinel-2 (TCI)")

        return f
