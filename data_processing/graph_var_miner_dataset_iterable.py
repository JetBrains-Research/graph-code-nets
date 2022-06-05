import gzip
import math
import multiprocessing
import pathlib
from itertools import chain
from typing import Iterator, Optional, List, Dict

import ijson
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Dataset

from data_processing.vocabulary.vocabulary import Vocabulary
from scripts.download_varnaming_dataset import download_from_google_drive

_graph_var_miner_edge_types = [
    "NextToken",
    "Child",
    "LastWrite",
    "LastUse",
    "ComputedFrom",
    "ReturnsTo",
    "FormalArgName",
    "GuardedBy",
    "GuardedByNegation",
    "LastLexicalUse",
]
_graph_var_miner_edge_types.extend(
    list(map(lambda s: f"reversed{s}", _graph_var_miner_edge_types))
)
_graph_var_miner_edge_types_to_idx = dict(
    (name, i) for i, name in enumerate(_graph_var_miner_edge_types)
)


class GraphVarMinerDatasetIterable(Dataset, IterableDataset):
    def __init__(
        self,
        config: dict,
        mode: str,
        vocabulary: Vocabulary,
        device="cpu",
        *,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        logger=None,
    ):
        print(f"New iterable dataset created! Id: {multiprocessing.current_process()}")
        self._config = config
        self._mode = mode
        self._vocabulary = vocabulary
        self.device = device
        self._logger = logger

        if "root" in self._config[self._mode]["dataset"]:
            self._root = self._config[self._mode]["dataset"]["root"]
        else:
            self._root = self._config["data"]["root"]

        if "cache_in_ram" in self._config[self._mode]["dataset"]:
            self._cache_in_ram = self._config[self._mode]["dataset"]["cache_in_ram"]
        elif "cache_in_ram" in self._config["data"]:
            self._cache_in_ram = self._config["data"]["cache_in_ram"]
        else:
            self._cache_in_ram = False

        if self._cache_in_ram:
            torch.multiprocessing.set_sharing_strategy("file_descriptor")

        if "preprocessed" in self._config[self._mode]["dataset"]:
            self._preprocessed = self._config[self._mode]["dataset"]["preprocessed"]
        else:
            self._preprocessed = self._config["data"]["preprocessed"]

        self._max_token_len = self._config["vocabulary"]["max_token_length"]
        self._debug = self._config[self._mode]["dataset"]["debug"]

        _raw_data_path = pathlib.Path(self._root, mode)
        if not _raw_data_path.exists():
            raise FileNotFoundError()

        self._data_files: list[pathlib.Path] = [
            f for f in _raw_data_path.iterdir() if f.is_file()
        ]

        self.__data_sample: Optional[Data] = None

        self._cached_in_ram: Dict[
            str, list
        ] = {}  # cache list of data samples, accessed by filename

        super().__init__(self._root, transform, pre_transform, pre_filter)

    def download(self):
        download_from_google_drive(self._root, self._config["data"]["link"])

    @property
    def raw_paths(self) -> List[str]:
        return [self._root]

    def _item_from_dict(self, dct) -> Data:
        nodes = list(dct["ContextGraph"]["NodeLabels"].values())
        tokens = self._process_tokens(nodes)
        marked_tokens = torch.tensor(
            np.array(nodes) == "<var>", dtype=torch.float, device=self.device
        )

        edge_index = []
        edge_attr = []
        for edges_typed_group in dct["ContextGraph"]["Edges"].items():
            edges_type = _graph_var_miner_edge_types_to_idx[edges_typed_group[0]]
            edge_index.extend(edges_typed_group[1])
            edge_attr.extend([edges_type] * len(edges_typed_group[1]))
        edge_index_t = torch.tensor(edge_index, device=self.device).t().contiguous()
        edge_weight_t = torch.tensor(
            edge_attr, dtype=torch.float, device=self.device
        )  # TODO incorrect, fix (must be edge_attr)

        filename = dct["filename"]
        name = self._process_tokens([dct["name"]])
        types = self._process_tokens(dct["types"])
        return Data(
            x=tokens,
            edge_index=edge_index_t,
            edge_weight=edge_weight_t,
            filename=filename,
            name=name,
            types=types,
            marked_tokens=marked_tokens,
        )

    def _process_tokens(self, tokens: list) -> torch.Tensor:
        out_tensor = np.zeros((len(tokens), self._max_token_len), dtype=int)
        for i, token in enumerate(tokens):
            if self._preprocessed:
                enc = token
            else:
                enc = self._vocabulary.encode(token)[: self._max_token_len]
            length = len(enc)
            out_tensor[i, :length] = enc
        return torch.Tensor(out_tensor, device=self.device)

    def _data_sample(self):
        if self.__data_sample is None:
            f = gzip.open(str(self._data_files[0]), "rb")
            items = ijson.items(f, "item")
            self.__data_sample = self._item_from_dict(next(items))
            f.close()

        return self.__data_sample

    @property
    def num_node_features(self) -> int:
        data = self._data_sample()
        if hasattr(data, "num_node_features"):
            return data.num_node_features
        raise AttributeError(
            f"'{data.__class__.__name__}' object has no "
            f"attribute 'num_node_features'"
        )

    @property
    def num_edge_features(self) -> int:
        data = self._data_sample()
        if hasattr(data, "num_edge_features"):
            return data.num_edge_features
        raise AttributeError(
            f"'{data.__class__.__name__}' object has no "
            f"attribute 'num_edge_features'"
        )

    def _items_from_file(self, filename):
        if self._cache_in_ram and filename in self._cached_in_ram:
            return iter(self._cached_in_ram[filename])
        f = gzip.open(str(filename), "rb")
        items = ijson.items(f, "item")
        items_iter = map(self._item_from_dict, items)
        if self._cache_in_ram:
            self._cached_in_ram[filename] = list(items_iter)

            # worker_num = torch.utils.data.get_worker_info().id
            # print(f"Worker #{worker_num}, cached {filename}")

            return iter(self._cached_in_ram[filename])
        else:
            return items_iter

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Data]:
        worker_info = torch.utils.data.get_worker_info()
        # print(f"New iterable created by worker {worker_info.id}! Id: {multiprocessing.current_process()}")
        if worker_info is None:
            files_slice = self._data_files
        else:
            per_worker = int(math.ceil(len(self._data_files) / worker_info.num_workers))
            worker_id = worker_info.id
            files_start = worker_id * per_worker
            files_end = min(len(self._data_files), files_start + per_worker)
            files_slice = self._data_files[files_start:files_end]
        return chain.from_iterable(map(self._items_from_file, files_slice))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
