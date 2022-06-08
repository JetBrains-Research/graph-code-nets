import sys

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger

from data_processing.graph_var_miner_dataloader import GraphVarMinerModule
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.vocabulary.spm_vocabulary import SPMVocabulary
from models.varnaming import VarNamingModel


def main():
    if len(sys.argv) < 2:
        raise ValueError("Expecting ckpt_path")

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
    model = VarNamingModel.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, vocabulary=vocabulary)

    trainer = pl.Trainer(logger=logger)

    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
