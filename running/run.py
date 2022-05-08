import pytorch_lightning as pl
from models.wraped_model import VarMisuseLayer
import yaml
from data_processing import vocabulary, geometric_graph_data_loader
from pytorch_lightning.loggers import WandbLogger
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("config_path")
args = ap.parse_args()

config_path = args.config_path
config = yaml.safe_load(open(config_path))
data_path = config["paths"]["data"]
vocabulary_path = config["paths"]["vocab"]

vocab = vocabulary.Vocabulary(vocab_path=vocabulary_path)
data = geometric_graph_data_loader.GraphDataModule(data_path, vocab, config)
data.prepare_data()
data.setup("fit")
# data.setup("test")
model = VarMisuseLayer(config, vocab.vocab_dim)
# wandb_logger = WandbLogger()
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=2, val_check_interval=0.1)
trainer.fit(
    model=model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)
