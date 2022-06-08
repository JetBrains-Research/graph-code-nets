import yaml
from data_processing import graph_data_loader, graph_dataset
from data_processing.vocabulary.great_vocabulary import GreatVocabulary

data_path = "data"
config_path = "config_var_misuse.yml"
vocabulary_path = "vocab.txt"

mode = "test"
config = yaml.safe_load(open(config_path))
vocab = GreatVocabulary(vocab_path="vocab.txt")

dataset = graph_dataset.GraphDataset(data_path, vocab, config, mode, debug=False)
dl = graph_data_loader.GraphDataModule(data_path, vocab, config)
dl.prepare_data()
dl.setup("fit")

for batch in dl.train_dataloader():
    print(batch)

for sample in dataset:  # type: ignore[attr-defined]
    print(sample, sep="\n")
