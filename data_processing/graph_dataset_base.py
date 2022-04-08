from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from typing import Any, TypeVar

from torch.utils.data import Dataset

T_E = TypeVar('T_E', bound=EnumMeta)


class GraphDatasetItemBase(ABC):
    def __init__(self, nodes: list[Any], edges: list[list[Enum, int, int]]):
        self.nodes = nodes
        self.edges = edges

    @abstractmethod
    def get_edges_types(self) -> T_E:
        ...


T_G = TypeVar('T_G', bound=GraphDatasetItemBase)


class GraphDatasetBase(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> T_G:
        ...
