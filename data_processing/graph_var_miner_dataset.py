import os
from enum import Enum, auto
from typing import Any

from data_processing.graph_dataset_base import GraphDatasetBase, GraphDatasetItemBase, T_E, T_G


class GraphVarMinerEdgeType(Enum):
    NEXT_TOKEN = auto()
    AST_CHILD = auto()
    LAST_WRITE = auto()
    LAST_USE = auto()
    COMPUTED_FROM = auto()
    RETURNS_TO = auto()
    FORMAL_ARG_NAME = auto()
    GUARDED_BY = auto()
    GUARDED_BY_NEGATION = auto()
    LAST_LEXICAL_USE = auto()
    ASSIGNABLE_TO = auto()
    ASSOCIATED_TOKEN = auto()
    HAS_TYPE = auto()
    ASSOCIATED_SYMBOL = auto()


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

    def get_edges_types(self) -> T_E:
        return GraphVarMinerEdgeType


class GraphVarMinerDataset(GraphDatasetBase):
    def __init__(self, data_path: str, mode: str, *, debug=False):
        self._data_path = os.path.join(data_path, mode)
        self._mode = mode
        self._data_files = os.listdir(self._data_path)

        # f = gzip.open("C:/Users/Urass/AppData/Roaming/JetBrains/PyCharm2021.3/scratches/scratch_5.json.gz", "r")
        # objs = ijson.items(f, 'item')
        # for i, obj in enumerate(objs):
        #     print(i, obj)


    def __getitem__(self, index: int) -> T_G:
        return GraphVarMinerItem([], [])
