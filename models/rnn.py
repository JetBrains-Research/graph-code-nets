import pytorch_lightning as pl
import torch


class RNN(pl.LightningModule):

    def __init__(self, model_config, shared_embedding=None, vocab_dim=None):
        super().__init__()
        self.config = model_config
