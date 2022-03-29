import os
import random

import json

import torch
from torch.utils.data import Dataset

import numpy as np

EDGE_TYPES = {
    'enum_CFG_NEXT': 0,
    'enum_LAST_READ': 1,
    'enum_LAST_WRITE': 2,
    'enum_COMPUTED_FROM': 3,
    'enum_RETURNS_TO': 4,
    'enum_FORMAL_ARG_NAME': 5,
    'enum_FIELD': 6,
    'enum_SYNTAX': 7,
    'enum_NEXT_SYNTAX': 8,
    'enum_LAST_LEXICAL_USE': 9,
    'enum_CALLS': 10
}


class GraphDataset(Dataset):

    def __init__(self, data_path, vocabulary, config, mode, debug=False):
        self.data_path = os.path.join(data_path, mode)
        self.vocabulary = vocabulary
        self.config = config
        self.mode = mode
        self.lines_data = list()
        files = os.listdir(self.data_path)
        if not debug:
            random.shuffle(files)
        for filename in files:
            with open(os.path.join(self.data_path, filename), 'r') as f:
                self.lines_data.extend(f.readlines())
        self.length = len(self.lines_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.process_line(self.lines_data[index])

    def to_raw_sample(self, json_data):

        def parse_edges(edges):
            relations = [[2 * EDGE_TYPES[rel[3]], rel[0], rel[1]] for rel in edges if rel[
                3] in EDGE_TYPES]
            relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]
            return relations

        tokens = [self.vocabulary.translate(t)[:self.config["data"]["max_token_length"]] for t in
                  json_data["source_tokens"]]
        edges = parse_edges(json_data["edges"])
        error_location = json_data["error_location"]
        repair_targets = json_data["repair_targets"]
        repair_candidates = [t for t in json_data["repair_candidates"] if isinstance(t, int)]
        return tokens, edges, error_location, repair_targets, repair_candidates

    def process_tokens(self, tokens):
        tokens = list(map(lambda x: list(np.pad(x, (0, self.config["data"]["max_token_length"] - len(x)))), tokens))
        return torch.Tensor(tokens)

    def process_line(self, line):
        tokens, edges, error_location, repair_targets, repair_candidates = self.to_raw_sample(json.loads(line))
        return self.process_tokens(tokens), edges, error_location, repair_targets, repair_candidates
