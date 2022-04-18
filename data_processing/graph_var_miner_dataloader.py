import os
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from data_processing.graph_var_miner_dataset import GraphVarMinerDataset


# DataLoader expects output to be tensor, but yet they are GraphDatasetItemBase, so we need collate_fn
def identity(x):
    return x


class GraphVarMinerModule(pl.LightningDataModule):
    def __init__(self, data_path: str):
        super().__init__()
        self._data_path = os.path.join(data_path)
        self._train: Dataset[Any]
        self._val: Dataset[Any]
        self._test: Dataset[Any]

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self._train = GraphVarMinerDataset(data_path=self._data_path, mode="train")
            self._val = GraphVarMinerDataset(
                data_path=self._data_path, mode="validation"
            )

        if stage == "test" or stage is None:
            self._test = GraphVarMinerDataset(data_path=self._data_path, mode="test")

    # shuffle is not supported due to IterableDataset
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=64, collate_fn=identity, num_workers=2
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=64, collate_fn=identity, num_workers=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=64, collate_fn=identity, num_workers=2)
