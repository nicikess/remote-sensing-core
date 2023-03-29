import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
from torch.utils.data import Dataset

from remote_sensing_core.constants import (
    Bands,
    S1_IMG_KEY,
    S2_IMG_KEY,
    LABEL_KEY,
    NUMPY_DTYPE,
)


class BenGeS(Dataset):
    def __init__(
        self,
        data_index_path: str,
        esaworldcover_index_path: str,
        root_dir_s1: str,
        root_dir_s2: str,
        number_of_classes: int,
        bands=Bands.RGB,
        transform=None,
        normalization_value=10_000,
    ):
        self.root_dir_s1 = root_dir_s1
        self.root_dir_s2 = root_dir_s2
        self.number_of_classes = number_of_classes
        self.bands = bands
        self.transform = transform
        self.normalization_value = normalization_value

        self.data_index = pd.read_csv(
            data_index_path
        )
        self.esaworldcover_index = pd.read_csv(
            esaworldcover_index_path
        )
        # TODO implement dataset split

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):

        # Sentinel 1
        file_name_s1 = self.data_index.loc[:, "patch_id_s1"][idx]
        path_image_s1 = os.path.join(self.root_dir_s1, file_name_s1) + "_all_bands.npy"
        img_s1 = np.load(path_image_s1)

        # Sentinel 2
        file_name_s2 = self.data_index.loc[:, "patch_id"][idx]
        path_image_s2 = os.path.join(self.root_dir_s2, file_name_s2) + "_all_bands.npy"
        img_s2 = np.load(path_image_s2)

        # Encode label
        threshold = 0.3
        label_vector = self.esaworldcover_index.loc[
            self.esaworldcover_index["patch_id"] == file_name_s2
        ]
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
        if self.bands == Bands.INFRARED:
            img_s2 = img_s2[[7, 3, 2, 1], :, :]
        if self.bands == Bands.ALL:
            img_s2 = img_s2

        # change type of img
        img_s1 = img_s1.astype(NUMPY_DTYPE)
        img_s2 = img_s2.astype(NUMPY_DTYPE)
        # Normalize img
        img_s1_normalized = img_s1 / self.normalization_value
        img_s2_normalized = img_s2 / self.normalization_value

        if self.transform:
            img_s1_normalized = self.transform(img_s1_normalized)
            img_s2_normalized = self.transform(img_s2_normalized)

        # Define output tensor
        output_tensor = {
            S1_IMG_KEY: img_s1_normalized,
            S2_IMG_KEY: img_s2_normalized,
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
