import pytorch_lightning as pl
from running.wraped_model import VarMisuseLayer
import yaml
from data_processing import vocabulary, graph_data_loader

data_path = "/home/timav/jb/graph-code-nets/data"
config_path = "/home/timav/jb/graph-code-nets/config.yml"
vocabulary_path = "/home/timav/jb/graph-code-nets/vocab.txt"

config = yaml.safe_load(open(config_path))
vocab = vocabulary.Vocabulary(vocab_path=vocabulary_path)
data = graph_data_loader.GraphDataModule(data_path, vocab, config)
data.prepare_data()
data.setup("fit")
# data.setup("test")
model = VarMisuseLayer(config["model"], config["training"], vocab.vocab_dim)

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2, val_check_interval=0.2)
trainer.validate(verbose=False)
trainer.fit(
    model=model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)
# trainer.validate(model=model, dataloaders=data.val_dataloader())
# trainer.test(model=model, dataloaders=data.test_dataloader())
