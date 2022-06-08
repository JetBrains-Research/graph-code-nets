import datetime
import os.path
import sys
import time

import pytorch_lightning as pl

import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_processing.graph_var_miner_dataloader import GraphVarMinerModule
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.vocabulary.spm_vocabulary import SPMVocabulary
from models.varnaming import VarNamingModel

import wandb


def main():
    ckpt_path = None
    if len(sys.argv) == 2:
        ckpt_path = sys.argv[1]

    config_path = (
        "config_varnaming.yaml"
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = WandbLogger(**config["logger"])

    wandb.save(config_path, policy="now")

    if "path" not in config["vocabulary"]:
        raise ValueError(
            "You need to specify path to vocabulary. "
            "Possibly, you should launch create_spm_vocabulary.py on your data"
        )

    if config["vocabulary"]["type"] == "spm":
        vocabulary = SPMVocabulary(config["vocabulary"]["path"])
    elif config["vocabulary"]["type"] == "great":
        vocabulary = GreatVocabulary(config["vocabulary"]["path"])
    else:
        raise ValueError(f'Unknown vocabulary type: {config["vocabulary"]["type"]}')

    datamodule = GraphVarMinerModule(config, vocabulary, logger=logger)
    datamodule.setup("fit")

    if ckpt_path is None:
        model = VarNamingModel(config, vocabulary)
    else:
        model = VarNamingModel.load_from_checkpoint(
            checkpoint_path=ckpt_path, config=config, vocabulary=vocabulary
        )

    if ckpt_path is None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%y.%m.%d_%H:%M:%S"
        )
        checkpoint_dirpath = f'{config["checkpoint"]["dir"]}/{timestamp}'
    else:
        checkpoint_dirpath = os.path.dirname(ckpt_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        save_top_k=int(config["checkpoint"]["top_k"]),
        monitor="validation_loss",
    )

    trainer = pl.Trainer(
        **config["trainer"], callbacks=[checkpoint_callback], logger=logger
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("Best model: ", checkpoint_callback.best_model_path)
    print(f"Top {checkpoint_callback.save_top_k}: {checkpoint_callback.best_k_models}")


if __name__ == "__main__":
    main()
