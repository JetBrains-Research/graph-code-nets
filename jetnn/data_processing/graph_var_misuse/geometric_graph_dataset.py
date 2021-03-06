import os
import json
import torch
from torch_geometric.data import Dataset, Data
from jetnn.data_processing.vocabularies.great.great_vocabulary import GreatVocabulary
from enum import Enum
from commode_utils.filesystem import get_line_by_offset
from jetnn.data_processing.commode_utils_extension import (
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
    def __init__(
        self,
        data_path: str,
        vocabulary: GreatVocabulary,
        config: dict,
        mode: str,
        debug: bool = False,
    ):
        super().__init__()
        self._data_path = os.path.join(data_path, mode)
        self._vocabulary = vocabulary
        self._config = config
        self._mode = mode
        self._data_files = os.listdir(self._data_path)
        self._files_offsets = get_files_offsets(self._data_path, debug)
        self._pref_sum_lines = get_files_count_lines(self._data_path)
        self._length = self._pref_sum_lines[-1]

    def len(self) -> int:
        return self._length

    def get(self, index: int) -> Data:
        file_index, line_index = get_file_index(self._pref_sum_lines, index)
        file_offset = self._files_offsets[file_index][line_index]
        return self.process_line(
            get_line_by_offset(
                os.path.join(self._data_path, self._data_files[file_index]), file_offset
            )
        )

    def _parse_line(self, json_data: dict) -> Data:
        # "edges" in input file is list of [before_index, after_index, edge_type, edge_type_name]
        def _parse_edges(edges: list) -> tuple[torch.Tensor, torch.Tensor]:
            _edge_index = [[rel[0] for rel in edges], [rel[1] for rel in edges]]
            _edge_attr = [rel[2] for rel in edges]
            return torch.tensor(_edge_index), torch.tensor(_edge_attr)

        max_seq_length = self._config["data"]["max_sequence_length"]
        max_tok_length = self._config["data"]["max_token_length"]

        tokens = torch.zeros(max_seq_length, max_tok_length)
        for i, source_token in json_data["source_tokens"][:max_seq_length]:
            translated_tokens = self._vocabulary.translate(source_token)[
                :max_tok_length
            ]
            tokens[i, : len(translated_tokens)] = translated_tokens

        edges = [
            e
            for e in json_data["edges"]
            if (max(e[0], e[1]) < self._config["data"]["max_sequence_length"])
        ]
        edge_index, edge_attr = _parse_edges(edges)

        error_location = json_data["error_location"]
        if error_location >= self._config["data"]["max_sequence_length"]:
            error_location = 0

        repair_targets = list(
            filter(
                lambda x: x < self._config["data"]["max_sequence_length"],
                json_data["repair_targets"],
            )
        )

        repair_candidates = list(
            filter(
                lambda x: x < self._config["data"]["max_sequence_length"],
                [t for t in json_data["repair_candidates"] if isinstance(t, int)],
            )
        )

        error_location_labels = torch.zeros(
            self._config["data"]["max_sequence_length"], dtype=torch.float32
        ).scatter_(0, torch.tensor(error_location), 1.0)
        repair_targets_labels = torch.zeros(
            self._config["data"]["max_sequence_length"], dtype=torch.float32
        ).scatter_(0, torch.tensor(repair_targets), 1.0)
        repair_candidates_labels = torch.zeros(
            self._config["data"]["max_sequence_length"], dtype=torch.float32
        ).scatter_(0, torch.tensor(repair_candidates), 1.0)
        labels = torch.stack(
            [
                error_location_labels,
                repair_targets_labels,
                repair_candidates_labels,
            ],
            1,
        )
        return_data = Data(tokens, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        return return_data

    def process_line(self, line: str) -> Data:
        return self._parse_line(json.loads(line))
