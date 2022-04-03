import pytorch_lightning as pl
import os
from data_processing.graph_dataset import GraphDataset
from data_processing.vocabulary import Vocabulary
from torch.utils.data import DataLoader


class MyDataLoader(pl.LightningDataModule):

    def __init__(self, data_path: str, vocabulary: Vocabulary, config: object):
        super().__init__()
        self._data_path = os.path.join(data_path)
        self._vocabulary = vocabulary
        self._config = config
        self._train, self._val, self._test = None, None, None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self._train = GraphDataset(data_path=self._data_path, vocabulary=self._vocabulary, config=self._config,
                                       mode='train')
            self._val = GraphDataset(data_path=self._data_path, vocabulary=self._vocabulary, config=self._config,
                                     mode='dev')

        if stage == "test" or stage is None:
            self._test = GraphDataset(data_path=self._data_path, vocabulary=self._vocabulary, config=self._config,
                                      mode='eval')

    # it doesn't support batching right now, because of dataset ;(
    def train_dataloader(self):
        return DataLoader(self._train, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=1)
