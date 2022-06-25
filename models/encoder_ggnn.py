import pytorch_lightning as pl
import torch
from torch_geometric.nn.conv import GatedGraphConv


class EncoderGGNN(pl.LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self._ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.Tensor:
        return self._ggnn(x, edge_index)
