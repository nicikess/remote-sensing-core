from typing import Dict, Optional

import torch
from torch import nn

from torchvision import transforms
from torchvision.transforms import Pad


class CompositeImageTransform(nn.Module):
    def __init__(
        self,
        convert_from_numpy: bool = False,
        padding_parameters: Optional[Dict] = None,
    ):
        super().__init__()
        transforms_list = []
        # Convert NUmpy array as is to torch tensor
        if convert_from_numpy:
            transforms_list.append(torch.from_numpy)
        # Pad an image tensor
        if padding_parameters:
            transforms_list.append(Pad(**padding_parameters))
        self._transform = transforms.Compose(transforms_list)

    def forward(self, x):
        return self._transform(x)


if __name__ == "__main__":
    import numpy as np

    img = np.random.rand(3, 120, 120)
    print(img.shape)
    comp_trans = CompositeImageTransform(
        # padding_parameters={
        #     "padding": 4
        # }
    )
    print(comp_trans(img).shape)
