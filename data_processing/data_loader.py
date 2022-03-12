import os
import random

import json
from random import shuffle
from glob import glob

# Edge types to be used in the models, and their (renumbered) indices -- the data_processing files contain
# reserved indices for several edge types that do not occur for this problem (e.g. UNSPECIFIED)
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


class DataLoader():

    def __init__(self, data_path, data_config, vocabulary):
        self.data_path = data_path
        self.config = data_config
        self.vocabulary = vocabulary

    def get_data_path(self, mode):
        if mode == "train":
            return os.path.join(self.data_path, "train")
        elif mode == "dev":
            return os.path.join(self.data_path, "dev")
        elif mode == "eval":
            return os.path.join(self.data_path, "eval")
        else:
            raise ValueError("Mode % not supported for batching; please use \"train\", \"dev\", or \"eval\".")

    def to_sample(self, json_data):

        def parse_edges(edges):
            # Reorder edges to [edge type, source, target] and double edge type index to allow reverse edges
            relations = [[2 * EDGE_TYPES[rel[3]], rel[0], rel[1]] for rel in edges if rel[
                3] in EDGE_TYPES]  # Note: we reindex edge types to be 0-based and filter unsupported edge types (
            # useful for ablations)
            relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]  # Add reverse edges
            return relations

        tokens = [self.vocabulary.translate(t)[:self.config["max_token_length"]] for t in json_data["source_tokens"]]
        edges = parse_edges(json_data["edges"])
        error_location = json_data["error_location"]
        repair_targets = json_data["repair_targets"]
        repair_candidates = [t for t in json_data["repair_candidates"] if isinstance(t, int)]
        return tokens, edges, error_location, repair_targets, repair_candidates


