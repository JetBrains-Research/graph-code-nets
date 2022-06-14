import pytorch_lightning as pl
import os
from data_processing.graph_var_misuse.geometric_graph_dataset import GraphDataset
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from torch_geometric.loader import DataLoader


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, vocabulary: GreatVocabulary, config: dict):
        super().__init__()
        self._data_path = os.path.join(data_path)
        self._vocabulary = vocabulary
        self._config = config
        self._train: GraphDataset
        self._val: GraphDataset
        self._test: GraphDataset

    def prepare_data(self):
        pass

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self._train = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="processed_train",
            )
            self._val = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="processed_dev",
            )

        if stage == "test" or stage is None:
            self._test = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="processed_eval",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self._config["data"]["batch_size"], num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self._config["data"]["batch_size"], num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test, batch_size=self._config["data"]["batch_size"], num_workers=8,
        )
