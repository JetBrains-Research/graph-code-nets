import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data_processing.graph_dataset import GraphDataset
from data_processing.vocabulary.vocabulary import Vocabulary


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, vocabulary: Vocabulary, config: dict):
        super().__init__()
        self._data_path = os.path.join(data_path)
        self._vocabulary = vocabulary
        self._config = config
        self._train: GraphDataset
        self._val: GraphDataset
        self._test: GraphDataset

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self._train = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="dev",
            )
            self._val = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="dev",
            )

        if stage == "test" or stage is None:
            self._test = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="eval",
            )

    # it doesn't support batching right now, because of dataset ;(
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=1)
