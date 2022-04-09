import ast
import gzip
import math
import os
from enum import Enum
from itertools import chain
from typing import Any, Iterator

import ijson
import torch
from torch.utils.data import IterableDataset

from data_processing.graph_dataset_base import GraphDatasetBase, GraphDatasetItemBase, T_E, T_G

_graph_var_miner_edge_types = ['NextToken',
                               'Child',
                               'LastWrite',
                               'LastUse',
                               'ComputedFrom',
                               'ReturnsTo',
                               'FormalArgName',
                               'GuardedBy',
                               'GuardedByNegation',
                               'LastLexicalUse']
_graph_var_miner_edge_types.extend(list(map(lambda s: f"reversed{s}", _graph_var_miner_edge_types)))

GraphVarMinerEdgeType = Enum('GraphVarMinerEdgeType', _graph_var_miner_edge_types)


class GraphVarMinerItem(GraphDatasetItemBase):
    def __init__(self,
                 filename: str,
                 nodes: list[Any],
                 edges: list[list[Enum, int, int]],
                 name: str,
                 types: list[str],
                 span: tuple[int, int]):
        super().__init__(nodes, edges)
        self.filename = filename
        self.name = name  # name of the observed variable
        self.types = types
        self.span = span

    @classmethod
    def from_dict(cls, dct):
        nodes = list(dct['ContextGraph']['NodeLabels'].values())  # ignoring indices
        edges = []
        for edges_typed_group in dct['ContextGraph']['Edges'].items():
            edges_type = GraphVarMinerEdgeType[edges_typed_group[0]]
            edges.extend(map(lambda l: [edges_type] + l, edges_typed_group[1]))

        return cls(dct['filename'],
                   nodes,
                   edges,
                   dct['name'],
                   dct['types'],
                   ast.literal_eval(dct['span']))

    def get_edges_types(self) -> T_E:
        return GraphVarMinerEdgeType


class GraphVarMinerDataset(GraphDatasetBase, IterableDataset):
    def __init__(self, data_path: str, mode: str, *, debug=False):
        self._data_path = os.path.join(data_path, mode)
        self._mode = mode
        self._data_files = os.listdir(self._data_path)
        self.debug = debug

    def _items_from_file(self, filename):
        if self.debug:
            print(f'Hi i am worker #{torch.utils.data.get_worker_info().id}, reading {filename}')
        full_filename = os.path.join(self._data_path, filename)
        f = gzip.open(full_filename, 'r')
        return map(GraphVarMinerItem.from_dict, ijson.items(f, 'item'))

    # might be implemented, but might be very inefficient due to
    # gzip streaming (__getitem__ requires random access) and json decoding
    def __getitem__(self, index: int) -> T_G:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T_G]:
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
