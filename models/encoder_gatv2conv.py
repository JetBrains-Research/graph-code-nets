import pytorch_lightning as pl
from torch.nn import ReLU
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import TransformerConv
from torch import Tensor
import torch


class EncoderGATv2Conv(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_attr_dim) -> None:
        super().__init__()
        modules = []
        for i in range(num_layers):
            in_channels_ = in_channels if i == 0 else hidden_channels
            modules.append(
                (
                    TransformerConv(
                        in_channels_, hidden_channels, edge_dim=edge_attr_dim
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
            modules.append(ReLU(inplace=True))

        self.model = Sequential("x, edge_index, edge_attr", modules)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tensor:  # type: ignore
        return self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
