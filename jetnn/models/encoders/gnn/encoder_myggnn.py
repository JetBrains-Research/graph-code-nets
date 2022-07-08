import pytorch_lightning as pl
import torch
from jetnn.models.encoders.gnn.custom_convs import typed_ggnn


class TypedGGNNEncoder(pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self._hidden_dim = model_config["hidden_dim"]
        self._num_layers = model_config["num_layers"]
        self._ggnn = GGNNTypedEdges.TypedGGNNConv(
            out_channels=self._hidden_dim,
            num_layers=self._num_layers,
            edge_dim=model_config["edge_attr_dim"],
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self._ggnn(x, edge_index, edge_attr=edge_attr)
