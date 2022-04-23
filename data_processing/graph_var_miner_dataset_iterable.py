import gzip
import gzip
import math
import pathlib
from itertools import chain
from typing import Iterator

import ijson
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Dataset

from data_processing.vocabulary import Vocabulary

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
_graph_var_miner_edge_types = dict((name, i) for i, name in enumerate(_graph_var_miner_edge_types))


# see https://github.com/python/mypy/issues/5317
# GraphVarMinerEdgeType = Enum("GraphVarMinerEdgeType", _graph_var_miner_edge_types)  # type: ignore


class GraphVarMinerDatasetIterable(Dataset, IterableDataset):
    def __init__(
            self,
            root: str,
            mode: str,
            vocabulary: Vocabulary,
            max_token_len: int = 10,
            *,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            debug=False,
    ):
        self._mode = mode
        self._vocabulary = vocabulary
        self._max_token_len = max_token_len

        self._raw_data_path = pathlib.Path(root, mode)
        if not self._raw_data_path.exists():
            raise FileNotFoundError()

        self._data_files: list[pathlib.Path] = [
            f for f in self._raw_data_path.iterdir() if f.is_file()
        ]
        self._debug = debug

        super().__init__(root, transform, pre_transform, pre_filter)

    def _item_from_dict(self, dct) -> Data:
        tokens = self._process_tokens(dct["ContextGraph"]["NodeLabels"].values())
        edge_index = []
        edge_attr = []
        for edges_typed_group in dct["ContextGraph"]["Edges"].items():
            edges_type = _graph_var_miner_edge_types[edges_typed_group[0]]
            edge_index.extend(edges_typed_group[1])
            edge_attr.extend([edges_type] * len(edges_typed_group[1]))
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr)

        filename = dct["filename"]
        name = dct["filename"]
        types = self._process_tokens(dct["types"])
        return Data(x=tokens,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    filename=filename,
                    name=name,
                    types=types)

    def _process_tokens(self, tokens: list) -> torch.Tensor:
        tokens = list(
            map(lambda x: list(np.pad(x, (0, self._max_token_len - len(x)))),
                map(lambda x: self._vocabulary.translate(x)[:self._max_token_len],
                    tokens)
                )
        )
        return torch.Tensor(tokens)

    def _items_from_file(self, filename):
        # if self.debug:
        #     print(
        #         f"Hi i am worker #{torch.utils.data.get_worker_info().id}, reading {filename}"
        #     )
        f = gzip.open(str(filename), "rb")
        return map(self._item_from_dict, ijson.items(f, "item"))

    def len(self) -> int:
        raise NotImplementedError

    def get(self, idx: int) -> Data:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Data]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_slice = self._data_files
        else:
            per_worker = int(math.ceil(len(self._data_files) / worker_info.num_workers))
            worker_id = worker_info.id
            files_start = worker_id * per_worker
            files_end = min(len(self._data_files), files_start + per_worker)
            files_slice = self._data_files[files_start:files_end]
        return chain.from_iterable(map(self._items_from_file, files_slice))

# class GraphVarMinerItem(torch_geometric.data.Dataset):
#     def __init__(
#         self,
#         filename: str,
#         nodes: list[Any],
#         edges: list[tuple[Enum, int, int]],
#         name: str,
#         types: list[str],
#         span: tuple[int, int],
#     ):
#         super().__init__(nodes, edges)
#         self.filename = filename
#         self.name = name  # name of the observed variable
#         self.types = types
#         self.span = span
#
#     @classmethod
#     def from_dict(cls, dct):
#         nodes = list(dct["ContextGraph"]["NodeLabels"].values())  # ignoring indices
#         edges = []
#         for edges_typed_group in dct["ContextGraph"]["Edges"].items():
#             edges_type = GraphVarMinerEdgeType[edges_typed_group[0]]
#             edges.extend(map(lambda l: (edges_type, l[0], l[1]), edges_typed_group[1]))
#
#         return cls(
#             dct["filename"],
#             nodes,
#             edges,
#             dct["name"],
#             dct["types"],
#             ast.literal_eval(dct["span"]),
#         )
#
#     def get_edges_types(self) -> EdgeType:
#         # mypy is unable to determine the type of dynamically created Enum
#         return GraphVarMinerEdgeType  # type: ignore
#
#
# class GraphVarMinerDataset(GraphDatasetBase, IterableDataset):
#     def __init__(self, data_path: str, mode: str, *, debug=False):
#         self._data_path = os.path.join(data_path, mode)
#         self._mode = mode
#         self._data_files = os.listdir(self._data_path)
#         self.debug = debug
#
#     def _items_from_file(self, filename):
#         if self.debug:
#             print(
#                 f"Hi i am worker #{torch.utils.data.get_worker_info().id}, reading {filename}"
#             )
#         full_filename = os.path.join(self._data_path, filename)
#         f = gzip.open(full_filename, "r")
#         return map(GraphVarMinerItem.from_dict, ijson.items(f, "item"))
#
#     # might be implemented, but might be very inefficient due to
#     # gzip streaming (__getitem__ requires random access) and json decoding
#     def __getitem__(self, index: int) -> GraphDatasetItem:
#         raise NotImplementedError
#
#     def __iter__(self) -> Iterator[GraphDatasetItem]:
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:
#             files_slice = self._data_files
#         else:
#             per_worker = int(math.ceil(len(self._data_files) / worker_info.num_workers))
#             worker_id = worker_info.id
#             files_start = worker_id * per_worker
#             files_end = min(len(self._data_files), files_start + per_worker)
#             files_slice = self._data_files[files_start:files_end]
#         return chain.from_iterable(map(self._items_from_file, files_slice))
