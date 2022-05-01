from typing import Any

import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import Tensor


class GCNEncoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, batch: Data) -> Tensor:  # type: ignore
        return self.gcn(batch.x, batch.edge_index, batch.edge_weight)
