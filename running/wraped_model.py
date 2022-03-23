import util
import torch
from models import rnn
import pytorch_lightning as pl


class VarMisuseLayer(pl.LightningModule):

    def __init__(self, config, vocab_dim):
        super().__init__()
        self.config = config
        self.vocab_dim = vocab_dim

        self.embedding = torch.normal(mean=0, std=self.config['base']['hidden_dim'] ** -0.5,
                                      size=[self.vocab_dim, self.config['base']['hidden_dim']])
        self.prediction = torch.nn.Linear(self.config['base']['hidden_dim'], 2)
        self.pos_enc = util.positional_encoding(self.config['base']['hidden_dim'], 5000)

        join_dicts = lambda d1, d2: {**d1, **d2}
        base_config = self.config['base']
        inner_model = self.config['configuration']
        if inner_model == 'rnn':
            self.model = rnn.RNN(join_dicts(base_config, self.config['rnn']), shared_embedding=self.embedding)
        else:
            raise ValueError('Unknown model component provided:', inner_model)
