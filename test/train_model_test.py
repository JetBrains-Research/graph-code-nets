import pytorch_lightning as pl
from running.wraped_model import VarMisuseLayer
import yaml
from data_processing import vocabulary, data_loader


data_path = '../data'
config_path = '../config.yml'
vocabulary_path = '../vocab.txt'
mode = 'test'

config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path=vocabulary_path)
data = data_loader.GraphDataset(data_path, vocab, config, mode, debug=False)
model = VarMisuseLayer(config['model'], config['training'], data.vocabulary.vocab_dim)

trainer = pl.Trainer(gpus=1)
trainer.fit(model=model, train_dataloaders=data)
