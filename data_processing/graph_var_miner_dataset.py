import bisect
import gzip
import json
import pathlib
from typing import Union, Any, Optional

import ijson
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

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


class GraphVarMinerDataset(Dataset):
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

        self._proc_data_path = pathlib.Path(root, f"{mode}_processed")
        self._proc_data_path.mkdir(parents=True, exist_ok=True)

        self._proc_cfg_data_path = self._proc_data_path.joinpath("info.json")
        self.__info_cfg: Optional[dict[str, Any]] = None
        self.__bisection_index: Optional[list[int]] = None

        self._data_files: list[pathlib.Path] = [
            f for f in self._raw_data_path.iterdir() if f.is_file()
        ]
        self._debug = debug

        super().__init__(root, transform, pre_transform, pre_filter)

    def _info_cfg(self) -> dict:
        if self.__info_cfg is None:
            with self._proc_cfg_data_path.open("r") as info_file:
                self.__info_cfg = json.load(info_file)
        return self.__info_cfg

    def _save_info_cfg(self, info_cfg):
        self.__info_cfg = info_cfg
        with self._proc_cfg_data_path.open("w") as info_file:
            json.dump(self.__info_cfg, info_file, indent=4)

    def _bisection_index(self) -> list[int]:
        if self.__bisection_index is None:
            info_cfg = self._info_cfg()
            self.__bisection_index = [0]
            for file in info_cfg["files"]:
                prev = self.__bisection_index[-1]
                self.__bisection_index.append(prev + file["count"])

        return self.__bisection_index

    def _filename_by_idx(self, idx) -> pathlib.Path:
        info_cfg = self._info_cfg()
        bisection_index = self._bisection_index()

        data_dir_idx = bisect.bisect_right(bisection_index, idx) - 1
        if len(info_cfg["files"]) < data_dir_idx:
            raise ValueError("Index out of range")

        data_dir_info = info_cfg["files"][data_dir_idx]
        data_dir_name = data_dir_info["name"]

        rel_item_idx = idx - bisection_index[data_dir_idx]
        rel_item_name = f"{rel_item_idx}.pt"

        return self._proc_data_path.joinpath(data_dir_name, rel_item_name)

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

    @property
    def raw_file_names(self) -> Union[str, list[str], tuple]:
        return []

    def download(self):
        return

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple]:
        return [str(self._proc_cfg_data_path)]

    def process(self):
        info_cfg = {
            "raw_data_path": str(self._raw_data_path),
            "proc_data_path": str(self._proc_data_path),
            "mode": self._mode,
            "total_count": 0,
            "files": [],
        }

        items_per_file = 5000
        with tqdm(total=len(self._data_files) * items_per_file) as pbar:
            for (i, data_file) in enumerate(self._data_files):
                info = {"name": str(data_file.name)}

                items = self._items_from_file(data_file)
                data_dir = self._proc_data_path.joinpath(data_file.name)
                data_dir.mkdir(parents=True, exist_ok=True)
                counter = 0
                for (j, item) in enumerate(items):
                    item_path = data_dir.joinpath(f"{j}.pt")
                    torch.save(item, str(item_path))
                    counter += 1
                    pbar.update(1)
                info["count"] = counter
                info_cfg["files"].append(info)
                info_cfg["total_count"] += counter
                print(f"{data_file} done!!!")

        self._save_info_cfg(info_cfg)

    def len(self) -> int:
        return self._info_cfg()["total_count"]

    def get(self, idx: int) -> Data:
        with self._filename_by_idx(idx).open("rb") as f:
            item = torch.load(f)
        return item

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
