import os
import random
import json
import torch
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from data_processing.vocabulary import Vocabulary
from enum import Enum


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

    def __init__(self, data_path: str, vocabulary: Vocabulary, config: object, mode: str, debug: bool = False):
        self._data_path = os.path.join(data_path, mode)
        self._vocabulary = vocabulary
        self._config = config
        self._mode = mode
        self._lines_data = list()
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
        return self.process_line(self._lines_data[index])

    def _to_raw_sample(self, json_data: dict):

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
        return tokens, edges, error_location, repair_targets, repair_candidates

    def _process_tokens(self, tokens: list) -> torch.Tensor:
        tokens = list(map(lambda x: list(np.pad(x, (0, self._config["data"]["max_token_length"] - len(x)))), tokens))
        return torch.Tensor(tokens)

    def process_line(self, line: str):
        tokens, edges, error_location, repair_targets, repair_candidates = self._to_raw_sample(json.loads(line))
        return self._process_tokens(tokens), edges, error_location, repair_targets, repair_candidates
