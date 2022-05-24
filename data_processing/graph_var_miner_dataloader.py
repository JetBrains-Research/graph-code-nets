from typing import Any, Optional

import pytorch_lightning as pl
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from data_processing.graph_var_miner_dataset import GraphVarMinerDataset

# DataLoader expects output to be tensor, but yet they are GraphDatasetItemBase, so we need collate_fn
from data_processing.graph_var_miner_dataset_iterable import (
    GraphVarMinerDatasetIterable,
)
from data_processing.vocabulary.vocabulary import Vocabulary


class GraphVarMinerModule(pl.LightningDataModule):
    def __init__(self, config: dict, vocabulary: Vocabulary):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self._train: Optional[Dataset[Any]] = None
        self._validation: Optional[Dataset[Any]] = None
        self._test: Optional[Dataset[Any]] = None

    def prepare_data(self):
        pass

    def _setup_dataset(self, mode: str):
        setattr(
            self,
            f"_{mode}",
            GraphVarMinerDatasetIterable(config=self._config, mode=mode, vocabulary=self._vocabulary),
        )

    def setup(self, stage: str = None):

        if stage == "fit" or stage is None:
            self._setup_dataset("train")
            self._setup_dataset("validation")

        if stage == "test" or stage is None:
            self._setup_dataset("test")

    def _get_dataloader(self, mode: str):
        dataset = getattr(self, f"_{mode}")
        dataloader_config = self._config["train"]["dataloader"]
        return DataLoader(
            dataset,
            batch_size=dataloader_config["batch_size"],
            num_workers=dataloader_config["num_workers"],
        )

    # shuffle is not supported due to IterableDataset
    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")
