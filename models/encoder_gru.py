import pytorch_lightning as pl
import torch


class EncoderGRU(pl.LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int, dropout_rate: int):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.rnn(states)
