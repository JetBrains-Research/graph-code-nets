import yaml
from data_processing import vocabulary, graph_dataset, data_loader


data_path = 'data'
config_path = 'config.yml'
vocabulary_path = 'vocab.txt'

mode = 'test'
config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path='vocab.txt')

dataset = graph_dataset.GraphDataset(data_path, vocab, config, mode, debug=False)
dl = data_loader.MyDataLoader(data_path, vocab, config)
dl.prepare_data()
dl.setup('fit')

for batch in dl.train_dataloader():
    print(batch)


for sample in dataset:
    tokens, edges, error_loc, repair_targets, repair_candidates = sample
    print(tokens, edges, error_loc, repair_targets, repair_candidates, sep='\n')
    break
