from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from typing import Any, TypeVar

from torch.utils.data import Dataset

EdgeType = TypeVar("EdgeType", bound=EnumMeta)


class GraphDatasetItemBase(ABC):
    def __init__(self, nodes: list[Any], edges: list[tuple[Enum, int, int]]):
        self.nodes = nodes
        self.edges = edges

    @abstractmethod
    def get_edges_types(self) -> EdgeType:
        ...


GraphDatasetItem = TypeVar("GraphDatasetItem", bound=GraphDatasetItemBase)


class GraphDatasetBase(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> GraphDatasetItem:
        ...
