import pytorch_lightning as pl
from running.wraped_model import VarMisuseLayer
import yaml
from data_processing import vocabulary, graph_data_loader

data_path = '../data'
config_path = '../config.yml'
vocabulary_path = '../vocab.txt'

config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path=vocabulary_path)
data = graph_data_loader.GraphDataModule(data_path, vocab, config)
data.prepare_data()
data.setup('fit')
model = VarMisuseLayer(config['model'], config['training'], vocab.vocab_dim)

trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=data.train_dataloader())
