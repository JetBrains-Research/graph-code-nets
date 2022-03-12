from data_processing import vocabulary, data_loader
import yaml

data_path = 'great'
config_path = 'config.yml'
vocabulary_path = 'vocab.txt'

mode = 'train'
config = yaml.safe_load(open(config_path))
vocabulary.Vocabulary(vocab_path='vocab.txt')
data = data_loader.DataLoader(data_path, config["data"], vocabulary.Vocabulary(vocabulary_path))

data_path = data.get_data_path(mode)
