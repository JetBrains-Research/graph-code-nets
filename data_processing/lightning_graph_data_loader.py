import pytorch_lightning as pl
import os
from data_processing.lightning_graph_dataset import GraphDataset
from data_processing.vocabulary import Vocabulary
from torch.utils.data import DataLoader
import torch
import numpy as np


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

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self._train = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="train_small",
            )
            self._val = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="dev_small",
            )

        if stage == "test" or stage is None:
            self._test = GraphDataset(
                data_path=self._data_path,
                vocabulary=self._vocabulary,
                config=self._config,
                mode="eval",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self._config["data"]["batch_size"],
            collate_fn=self._collate_fn,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self._config["data"]["batch_size"],
            collate_fn=self._collate_fn,
            num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self._config["data"]["batch_size"],
            collate_fn=self._collate_fn,
            num_workers=8,
        )

    def _collate_fn(self, batch) -> tuple:
        batch = [
            (
                e["tokens"],
                e["edges"],
                e["error_location"],
                e["repair_targets"],
                e["repair_candidates"],
            )
            for e in batch
        ]
        batch_dim = len(batch)
        batch = list(zip(*batch))
        # padding
        tokens = [
            list(
                map(
                    lambda x: list(
                        np.pad(
                            x, (0, self._config["data"]["max_token_length"] - len(x))
                        )
                    ),
                    y,
                )
            )
            for y in batch[0]
        ]
        new_tokens = np.zeros(
            (
                len(tokens),
                max([len(x) for x in tokens]),
                self._config["data"]["max_token_length"],
            ),
            dtype=int,
        )
        for i in range(len(tokens)):
            new_tokens[i][: len(tokens[i])] = tokens[i]
        token_tensor = torch.tensor(new_tokens)

        # adding batch dimension
        edge_batches = torch.tensor(
            np.repeat(np.arange(0, batch_dim), [len(edges) for edges in batch[1]])
        )
        edge_tensor = torch.tensor(np.concatenate(batch[1]))
        edge_tensor = torch.stack(
            [edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 2]],
            dim=1,
        )

        # simple constant list
        error_location = torch.tensor(batch[2])

        # also adding batch dimensions
        target_batches = torch.tensor(
            np.repeat(np.arange(0, batch_dim), [len(targets) for targets in batch[3]])
        )
        repair_targets = torch.tensor(np.concatenate(batch[3]))
        repair_targets = torch.stack([target_batches, repair_targets], dim=1)
        repair_targets = repair_targets.to(torch.long)

        candidates_batches = torch.tensor(
            np.repeat(
                np.arange(0, batch_dim), [len(candidates) for candidates in batch[4]]
            )
        )
        repair_candidates = torch.tensor(np.concatenate(batch[4]))
        repair_candidates = torch.stack([candidates_batches, repair_candidates], dim=1)
        return (
            token_tensor,
            edge_tensor,
            error_location,
            repair_targets,
            repair_candidates,
        )
