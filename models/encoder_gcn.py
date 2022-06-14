import pytorch_lightning as pl
from torch.nn import ReLU
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GCNConv
from torch import Tensor
import torch


class GCNEncoder(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers) -> None:
        super().__init__()
        modules = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else hidden_channels
            modules.append(
                (GCNConv(in_channels_, hidden_channels), "x, edge_index -> x")
            )
            modules.append(ReLU(inplace=True))

        self.model = Sequential("x, edge_index", modules)

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> Tensor:  # type: ignore
        return self.model(x, edge_index)
