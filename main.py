import json
import yaml
from data_processing import vocabulary, data_loader

import torch.utils.data.dataset

data_path = 'great'
config_path = 'config.yml'
vocabulary_path = 'vocab.txt'

mode = 'test'
config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path='vocab.txt')

dataset = data_loader.GraphDataset(data_path, vocab, config, mode, debug=False)

for sample in dataset:
    tokens, edges, error_loc, repair_targets, repair_candidates = sample
    print(tokens, edges, error_loc, repair_targets, repair_candidates, sep='\n')
    break
