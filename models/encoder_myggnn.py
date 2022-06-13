import pytorch_lightning as pl
import torch
from my_models import GGNNTypedEdges


class EncoderMyGGNN(pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self._hidden_dim = model_config["hidden_dim"]
        self._num_layers = model_config["num_layers"]
        self._ggnn = GGNNTypedEdges.MyGatedGraphConv(
            out_channels=self._hidden_dim, num_layers=self._num_layers, edge_dim=model_config["edge_attr_dim"]
        )

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_attr: torch.tensor) -> torch.Tensor:
        return self._ggnn(x, edge_index, edge_attr=edge_attr)
