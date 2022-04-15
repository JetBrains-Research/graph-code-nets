import yaml
from data_processing import vocabulary, graph_dataset, graph_data_loader


data_path = "data"
config_path = "config.yml"
vocabulary_path = "vocab.txt"

mode = 'train'
config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path="vocab.txt")

dataset = graph_dataset.GraphDataset(data_path, vocab, config, mode, debug=False)
dl = graph_data_loader.GraphDataModule(data_path, vocab, config)
dl.prepare_data()
dl.setup("fit")

for batch in dl.train_dataloader():
    for e in batch:
        print(e)
