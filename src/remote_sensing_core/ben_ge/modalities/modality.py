from typing import Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from torch import nn


class Modality(ABC):
    NAME = None

    def __init__(
        self,
        data_root_path: Union[str, Path],
        transform: Optional[nn.Module] = None,
        numpy_dtype: Optional[str] = None,
    ):
        self.data_root_path = Path(data_root_path)
        self.transform = transform
        self.numpy_dtype = numpy_dtype

    @abstractmethod
    def load_sample(self, patch_id, *args, **kwargs):
        pass

    def transform_sample(self, sample, *args, **kwargs):
        if self.numpy_dtype:
            sample = sample.astype(self.numpy_dtype)
        if self.transform:
            sample = self.transform(sample)
        return sample