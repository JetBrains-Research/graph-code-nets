import pytorch_lightning as pl
import os
from graph_dataset import GraphDataset
from torch.utils.data import DataLoader


class MyDataLoader(pl.LightningDataModule):

    def __init__(self, data_path, vocabulary, config):
        super().__init__()
        self.data_path = os.path.join(data_path)
        self.vocabulary = vocabulary
        self.config = config
        self.train, self.val, self.test = None, None, None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = GraphDataset(data_path=self.data_path, vocabulary=self.vocabulary, config=self.config,
                                      mode='train')
            self.val = GraphDataset(data_path=self.data_path, vocabulary=self.vocabulary, config=self.config,
                                    mode='dev')

        if stage == "test" or stage is None:
            self.test = GraphDataset(data_path=self.data_path, vocabulary=self.vocabulary, config=self.config,
                                     mode='eval')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)
