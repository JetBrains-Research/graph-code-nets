import pytorch_lightning as pl
import torch
from torch_geometric.nn.conv import GatedGraphConv


class GGNNEncoder(pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self._hidden_dim = model_config["hidden_dim"]
        self._num_layers = model_config["num_layers"]
        self._ggnn = GatedGraphConv(
            out_channels=self._hidden_dim, num_layers=self._num_layers
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self._ggnn(x, edge_index)
