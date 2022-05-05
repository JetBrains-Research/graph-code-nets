from abc import ABC, abstractmethod
from typing import Iterable


class Vocabulary(ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def encode(self, token: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, encoded: list[int]) -> str:
        ...

    @abstractmethod
    def pad_id(self) -> int:
        ...
