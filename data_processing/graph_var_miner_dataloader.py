import os
from typing import Any

import pytorch_lightning as pl
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from data_processing.graph_var_miner_dataset import GraphVarMinerDataset

# DataLoader expects output to be tensor, but yet they are GraphDatasetItemBase, so we need collate_fn
from data_processing.graph_var_miner_dataset_iterable import (
    GraphVarMinerDatasetIterable,
)
from data_processing.vocabulary import Vocabulary


class GraphVarMinerModule(pl.LightningDataModule):
    def __init__(self, root: str, vocabulary: Vocabulary, process=False):
        super().__init__()
        self._root = os.path.join(root)
        self._vocabulary = vocabulary
        self._process = process
        self._train: Dataset[Any]
        self._val: Dataset[Any]
        self._test: Dataset[Any]

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if self._process:
            cls = GraphVarMinerDataset
        else:
            cls = GraphVarMinerDatasetIterable

        if stage == "fit" or stage is None:
            self._train = cls(
                root=self._root, mode="train", vocabulary=self._vocabulary
            )
            self._val = cls(
                root=self._root, mode="validation", vocabulary=self._vocabulary
            )

        if stage == "test" or stage is None:
            self._test = cls(root=self._root, mode="test", vocabulary=self._vocabulary)

    # shuffle is not supported due to IterableDataset
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=64, num_workers=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=64, num_workers=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=64, num_workers=2)
