import pytorch_lightning as pl
import torch


class TwoPointerFCN(pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self.hidden_dim = model_config["hidden_dim"]
        self.layer = torch.nn.Linear(self.hidden_dim, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layer(states)
