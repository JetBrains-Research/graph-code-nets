import os
import random

import json
from random import shuffle
from glob import glob
from torch.utils.data import Dataset

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


class MyDataset(Dataset):

    def __init__(self, data_path, mode):
        self.data_path = os.path.join(data_path, mode)
        self.mode = mode
        self.lines_data = list()
        files = os.listdir(self.data_path)
        shuffle(files)
        for filename in files:
            with open(os.path.join(self.data_path, filename), 'r') as f:
                self.lines_data.extend(f.readlines())
        self.length = len(self.lines_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.lines_data[index]


class MainDataLoader:

    def __init__(self, data_path, data_config, vocabulary):
        self.data_path = data_path
        self.config = data_config
        self.vocabulary = vocabulary

    def batcher(self, mode="train"):
        dataset = MyDataset(self.data_path, mode)
        batches_gen = self.to_batch(dataset, mode)
        return batches_gen

    def get_data_path(self, mode):
        if mode in ("train", "dev", "eval"):
            return os.path.join(self.data_path, mode)
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

    def to_batch(self, sample_generator, mode):

        if isinstance(mode, bytes): mode = mode.decode('utf-8')

        def sample_len(sample):
            return len(sample[0])

        # Generates a batch with similarly-sized sequences for efficiency
        def make_batch(buffer):
            pivot = sample_len(random.choice(buffer))
            buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
            batch = []
            max_seq_len = 0
            for sample in buffer:
                max_seq_len = max(max_seq_len, sample_len(sample))
                if max_seq_len * (len(batch) + 1) > self.config['max_batch_size']:
                    break
                batch.append(sample)
            batch_dim = len(batch)
            buffer = buffer[batch_dim:]
            batch = list(zip(*batch))
            return buffer, (batch[0], batch[1], batch[2], batch[3], batch[4])


        # Keep samples in a buffer that is (ideally) much larger than the batch size to allow efficient batching
        buffer = []
        num_samples = 0

        for line in sample_generator:
            json_sample = json.loads(line)
            sample = self.to_sample(json_sample)
            if sample_len(sample) > self.config['max_sequence_length']:
                continue
            buffer.append(sample)
            num_samples += 1
            if mode == 'dev' and num_samples >= self.config['max_valid_samples']:
                break
            if sum(sample_len(sample) for _ in buffer) > self.config['max_buffer_size'] * self.config['max_batch_size']:
                buffer, batch = make_batch(buffer)
                yield batch
        # Drain the buffer upon completion
        while buffer:
            buffer, batch = make_batch(buffer)
            yield batch
