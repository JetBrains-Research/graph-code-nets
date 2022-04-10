import pytorch_lightning as pl
import torch


class RNN(pl.LightningModule):

    def __init__(self, model_config, shared_embedding=None, vocab_dim=None):
        super().__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout_rate = model_config['dropout_rate']
        self.layer1 = torch.nn.Linear(self.hidden_dim // 2, 1)
        self.layer2 = torch.nn.Linear(self.hidden_dim // 2, 1)

    def forward(self, states):
        # just two different linear layers, which have different input features
        return torch.stack([self.layer1(states[:, :, :self.hidden_dim // 2]).squeeze(),
                            self.layer2(states[:, :, self.hidden_dim // 2:self.hidden_dim]).squeeze()], dim=1)

