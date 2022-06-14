from typing import Any

import pytorch_lightning as pl
import torch_scatter
from torch.nn import ReLU, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential
from torch import Tensor


class GCNEncoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers) -> None:
        super().__init__()
        modules: list[Any] = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else hidden_channels
            modules.append(
                (GCNConv(in_channels_, hidden_channels), "x, edge_index -> x")
            )
            modules.append(ReLU(inplace=True))
        modules.append(Linear(hidden_channels, out_channels))

        self.model = Sequential("x, edge_index, edge_weight", modules)

    def forward(self, batch: Data) -> Tensor:  # type: ignore
        enc = self.model(batch.x, batch.edge_index, batch.edge_weight)
        return GCNEncoder._scatter_marked_nodes(enc, batch)

    @staticmethod
    def _scatter_marked_nodes(enc, batch):
        return torch_scatter.scatter_mean(
            enc * batch.marked_tokens.unsqueeze(-1), batch.batch, dim=0
        ).unsqueeze(1)
