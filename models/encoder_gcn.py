import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import ReLU, Linear
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GCNConv


class GCNEncoder(pl.LightningModule):
    def __init__(
        self, in_channels, hidden_channels, num_layers, out_channels=None
    ) -> None:
        super().__init__()
        modules = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else hidden_channels
            modules.append(
                (GCNConv(in_channels_, hidden_channels), "x, edge_index -> x")
            )
            modules.append(ReLU(inplace=True))
        if out_channels is not None:
            modules.append(Linear(hidden_channels, out_channels))

        self.model = Sequential("x, edge_index", modules)

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> Tensor:  # type: ignore
        return self.model(x, edge_index)
