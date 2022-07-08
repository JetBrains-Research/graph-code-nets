import torch
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.tasks.varmisuse import VarMisuseModel
import yaml
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.graph_var_misuse import geometric_graph_data_loader
from pytorch_lightning.loggers import WandbLogger
import argparse

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("config_path")
args = ap.parse_args()

config_path = args.config_path
config = yaml.safe_load(open(config_path))
data_path = config["paths"]["data"]
vocabulary_path = config["paths"]["vocab"]

vocab = GreatVocabulary(vocab_path=vocabulary_path)
data = geometric_graph_data_loader.GraphDataModule(data_path, vocab, config)
data.prepare_data()
data.setup("fit")
# data.setup("test")
model = VarMisuseModel(config, vocab.vocab_dim)

wandb_logger = WandbLogger(project="graph-nets-test")
checkpoint_callback = ModelCheckpoint(dirpath="checkpoint/varmisuse/")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=2,
    val_check_interval=0.2,
    logger=wandb_logger,
    accumulate_grad_batches=2,
    callbacks=[checkpoint_callback],
)
trainer.fit(
    model=model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)
