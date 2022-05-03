import os
import json
from typing import Any

from torch_geometric.data import Dataset
from data_processing.vocabulary import Vocabulary
from enum import Enum
from commode_utils.filesystem import get_line_by_offset
from data_processing.commode_utils_extension import (
    get_files_count_lines,
    get_files_offsets,
    get_file_index,
)


class EdgeTypes(Enum):
    enum_CFG_NEXT = 0
    enum_LAST_READ = 1
    enum_LAST_WRITE = 2
    enum_COMPUTED_FROM = 3
    enum_RETURNS_TO = 4
    enum_FORMAL_ARG_NAME = 5
    enum_FIELD = 6
    enum_SYNTAX = 7
    enum_NEXT_SYNTAX = 8
    enum_LAST_LEXICAL_USE = 9
    enum_CALLS = 10


class GraphDataset(Dataset):
    def __init__(self, data_path: str, vocabulary: Vocabulary, config: dict, mode: str, debug: bool = False):
        super().__init__()
        self._data_path = os.path.join(data_path, mode)
        self._vocabulary = vocabulary
        self._config = config
        self._mode = mode
        self._data_files = os.listdir(self._data_path)
        self._files_offsets = get_files_offsets(self._data_path, debug)
        self._pref_sum_lines = get_files_count_lines(self._data_path)
        self._length = self._pref_sum_lines[-1]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, line_index = get_file_index(self._pref_sum_lines, index)
        file_offset = self._files_offsets[file_index][line_index]
        return self.process_line(
            get_line_by_offset(
                os.path.join(self._data_path, self._data_files[file_index]), file_offset
            )
        )

    def _parse_line(self, json_data: dict) -> dict[str, Any]:

        # "edges" in input file is list of [before_index, after_index, edge_type, edge_type_name]
        def _parse_edges(edges: list):
            _edge_index = [[rel[0], rel[1]] for rel in edges]
            _edge_attr = [rel[2] for rel in edges]
            return _edge_index, _edge_attr

        tokens = [
            self._vocabulary.translate(t)[: self._config["data"]["max_token_length"]]
            for t in json_data["source_tokens"]
        ]
        edge_index, edge_attr = _parse_edges(json_data["edges"])
        error_location = json_data["error_location"]
        repair_targets = json_data["repair_targets"]
        repair_candidates = [
            t for t in json_data["repair_candidates"] if isinstance(t, int)
        ]
        return {
            "edge_index": edge_index,
            "tokens": tokens,
            "error_location": error_location,
            "repair_targets": repair_targets,
            "repair_candidates": repair_candidates,
        }

    def process_line(self, line: str) -> dict[str, Any]:
        return self._parse_line(json.loads(line))
