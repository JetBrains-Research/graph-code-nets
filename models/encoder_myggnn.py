import pytorch_lightning as pl
import torch
from my_models import GGNNTypedEdges


class EncoderMyGGNN(pl.LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int, edge_attr_dim: int):
        super().__init__()
        self._ggnn = GGNNTypedEdges.MyGatedGraphConv(
            out_channels=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_attr_dim,
        )

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor
    ) -> torch.Tensor:
        return self._ggnn(x, edge_index, edge_attr=edge_attr)
