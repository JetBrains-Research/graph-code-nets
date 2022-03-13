import json
import yaml
from data_processing import vocabulary, data_loader

import torch.utils.data.dataset

data_path = 'great'
config_path = 'config.yml'
vocabulary_path = 'vocab.txt'

mode = 'train'
config = yaml.safe_load(open(config_path))
vocabulary.Vocabulary(vocab_path='vocab.txt')
data = data_loader.MainDataLoader(data_path, config["data"], vocabulary.Vocabulary(vocabulary_path))

# dataset = torch.utils.data.DataLoader(data_path + '/train/train__VARIABLE_MISUSE__SStuB.txt-00000-of-00300')

batch_gen = data.batcher('test')
for elem in batch_gen:
    print(elem)
    break
