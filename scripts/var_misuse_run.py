import os

import torch
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from models.geometric_wrapped_model import VarMisuseLayer
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
print(config["mode"])
print(config["trainer"])
data.setup(config["mode"])
model = VarMisuseLayer(config, vocab.vocab_dim)

wandb_logger = WandbLogger(**config["logger"])
checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoint/varmisuse_{config['model_name']}/")

trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    **config["trainer"]
)
trainer.fit(
    model=model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)

if config["mode"] == "holdout":
    all_losses = []
    print("Running holdout loss computation")
    for batch in tqdm(data.test_dataloader()):
        with torch.no_grad():
            loc_loss, target_loss = model.retrieve_per_sample_losses(batch)
            total_loss = loc_loss + target_loss
            all_losses.extend(total_loss)

    np.save(config["paths"]["holdout_losses"], np.array(all_losses))
