import pytorch_lightning as pl
import torch


class EncoderGRU(pl.LightningModule):
    def __init__(self, model_config: dict):
        super().__init__()
        self.hidden_dim = model_config["hidden_dim"]
        self.num_layers = model_config["num_layers"]
        self.dropout_rate = model_config["dropout_rate"]
        self.rnn = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True,
        )

    def forward(self, states: torch.tensor) -> torch.tensor:
        return self.rnn(states)
