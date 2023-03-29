import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
import torch
from torchvision import transforms
import pandas as pd

# set random seeds
torch.manual_seed(42)
np.random.seed(42)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        sample["data_S2"] = torch.from_numpy(sample["data_S2"])

        return sample


class Normalize(object):
    """Normalize data."""

    def __init__(self):
        """
        :param size: edge length of quadratic output size
        """

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        # normalize S2 data by dividing by 10k
        sample["data_S2"] = sample["data_S2"] / 10000

        return sample


class Data:
    """Dataset containing only S2 data."""

    def __init__(
        self,
        dataroot_S2="s2_npy/",  # root directory where S2 data resides
        labels_threshold=0.05,  # each label considered must cover at least this fraction of pixel in each image
        labels_n=3,  # will only keep this number of labels, in order of decreasing coverage fraction (None == all classes)
        transform=transforms.Compose([Normalize()]),
    ):

        self.transform = transform
        self.labels_threshold = labels_threshold
        self.labels_n = labels_n

        self.datafiles = []
        self.labels = pd.read_csv(
            os.path.join(dataroot_S2, "ben-ge-s_esaworldcover.csv"),
            index_col="patch_id",
        )

        # read in all data files
        for filename in os.listdir(dataroot_S2):
            if not filename.endswith("_all_bands.npy"):
                continue
            self.datafiles.append(os.path.join(dataroot_S2, filename))

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):

        data = np.load(self.datafiles[idx])
        tile = os.path.split(self.datafiles[idx])[-1].replace("_all_bands.npy", "")
        label_fractions = self.labels.loc[tile].loc["tree_cover":"moss_and_lichen"]

        # assemble labels - one-hot encoding
        labels_onehot = [l >= self.labels_threshold for l in label_fractions.values]

        # assemble labels - ranked list
        label_fractions2 = label_fractions.loc[
            label_fractions >= self.labels_threshold
        ]  # apply threshold
        labels_ranked = label_fractions2.index[
            np.argsort(label_fractions2.values)[::-1]
        ][: min(len(label_fractions2), self.labels_n)].values

        sample = {
            "idx": idx,
            "labels_onehot": labels_onehot,
            "labels_ranked": labels_ranked,
            "labels_raw": label_fractions,
            "tile": os.path.split(self.datafiles[idx])[-1].replace(
                "_all_bands.npy", ""
            ),
            "data_S2": data,
            "imgfile": self.datafiles[idx],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def display(self, idx):
        sample = self[idx]

        imgdata = sample["data_S2"]
        imgdata = np.dstack([imgdata[3], imgdata[2], imgdata[1]])

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(
            1.3
            * (imgdata - np.min(imgdata, axis=(0, 1)))
            / (np.max(imgdata, axis=(0, 1)) - np.min(imgdata, axis=(0, 1)))
            + 0.1
        )
        ax.set_title("Sentinel-2 (TCI)")

        return f


# how to get labels?
a = Data()
i = 100
print(a[i]["labels_raw"])
print(a[i]["labels_ranked"])
print(a[i]["labels_onehot"])
