from typing import Any

import torch
from torch import nn
import pytorch_lightning as pl


class LitDownstream(pl.LightningModule):
    def __init__(
        self,
        data_key: str,
        label_key: str,
        downstream_model: nn.Module,
        loss: nn.Module,
        learning_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_key = data_key
        self.label_key = label_key

        self.downstream_model = downstream_model
        self.learning_rate = learning_rate
        self.loss = loss

    def training_step(self, batch, batch_index, *args: Any, **kwargs: Any):
        x = batch[self.data_key]
        y_hat = self.downstream_model(x)
        y = batch[self.label_key]
        self.log()
        return self.loss(y_hat, y)

    def configure_optimizers(self) -> Any:
        params = list(self.downstream_model.parameters())
        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        return {
            "optimizer": opt,
        }


if __name__ == '__main__':
    batch = {
        "s2_img": torch.randn(11, 120, 120),
        "single_label": 4
    }
