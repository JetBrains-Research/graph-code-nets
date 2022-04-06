import os
import random
import json
import torch
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from data_processing.vocabulary import Vocabulary
from enum import Enum
from commode_utils.filesystem import get_line_by_offset
from data_processing.commde_utils_extension import get_files_count_lines, get_files_offsets, get_file_index


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
    _return_list = namedtuple('return_list',
                              ['tokens', 'edges', 'error_location', 'repair_targets', 'repair_candidates'])

    def __init__(self, data_path: str, vocabulary: Vocabulary, config: object, mode: str, debug: bool = False):
        self._data_path = os.path.join(data_path, mode)
        self._vocabulary = vocabulary
        self._config = config
        self._mode = mode
        self._lines_data = list()
        self._data_files = os.listdir(self._data_path)
        self._files_offsets = get_files_offsets(self._data_path)
        self._count_lines = get_files_count_lines(self._data_path)
        files = os.listdir(self._data_path)
        if not debug:
            random.shuffle(files)
        for filename in files:
            with open(os.path.join(self._data_path, filename), 'r') as f:
                self._lines_data.extend(f.readlines())
        self.length = len(self._lines_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        file_index, line_index = get_file_index(self._count_lines, index)
        file_offset = self._files_offsets[file_index][line_index]
        return self.process_line(get_line_by_offset(os.path.join(self._data_path, self._data_files[file_index]), file_offset))

    def _to_raw_sample(self, json_data: dict) -> namedtuple:

        # edges is list of [before_index, after_index, edge_type, edge_type_name]
        def parse_edges(edges: list):
            relations = [
                [2 * EdgeTypes[rel[3]].value, rel[0], rel[1]]
                for rel in edges
            ]
            relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]
            return relations

        tokens = [
            self._vocabulary.translate(t)[:self._config["data"]["max_token_length"]]
            for t in json_data["source_tokens"]
        ]
        edges = parse_edges(json_data["edges"])
        error_location = json_data["error_location"]
        repair_targets = json_data["repair_targets"]
        repair_candidates = [t for t in json_data["repair_candidates"] if isinstance(t, int)]
        return self._return_list(tokens, edges, error_location, repair_targets, repair_candidates)

    def _process_tokens(self, tokens: list) -> torch.Tensor:
        tokens = list(map(lambda x: list(np.pad(x, (0, self._config["data"]["max_token_length"] - len(x)))), tokens))
        return torch.Tensor(tokens)

    def process_line(self, line: str) -> namedtuple:
        return_values = self._to_raw_sample(json.loads(line))
        tokens, edges, error_location, repair_targets, repair_candidates = \
            return_values.tokens, return_values.edges, return_values.error_location, return_values.repair_targets, \
            return_values.repair_candidates
        return self._return_list(self._process_tokens(tokens), edges, error_location, repair_targets, repair_candidates)
