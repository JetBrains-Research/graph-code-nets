import sys

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger

from jetnn.data_processing.graph_var_miner.graph_var_miner_dataloader import (
    GraphVarMinerModule,
)
from jetnn.data_processing.vocabularies.great.great_vocabulary import GreatVocabulary
from jetnn.data_processing.vocabularies.spm.spm_vocabulary import SPMVocabulary
from jetnn.models import VarNamingModel


def main():
    if len(sys.argv) < 3:
        raise ValueError("Expecting test/validate and ckpt_path")

    test_or_validation = sys.argv[1]
    if test_or_validation not in ["test", "validation"]:
        raise ValueError(f"Unexpected test_or_validate: {test_or_validation}")

    ckpt_path = sys.argv[2]

    config_path = "config_varnaming.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = WandbLogger(**config["logger"])

    wandb.save(config_path, policy="now")

    if "path" not in config["vocabularies"]:
        raise ValueError(
            "You need to specify path to vocabularies. "
            "Possibly, you should launch create_spm_vocabulary.py on your data"
        )

    if config["vocabularies"]["type"] == "spm":
        vocabulary = SPMVocabulary(config["vocabularies"]["path"])
    elif config["vocabularies"]["type"] == "great":
        vocabulary = GreatVocabulary(config["vocabularies"]["path"])
    else:
        raise ValueError(f'Unknown vocabularies type: {config["vocabularies"]["type"]}')

    datamodule = GraphVarMinerModule(config, vocabulary, logger=logger)
    if test_or_validation == "test":
        datamodule.setup("test")
    else:
        datamodule.setup("fit")

    model = VarNamingModel.load_from_checkpoint(
        checkpoint_path=ckpt_path, config=config, vocabulary=vocabulary
    )

    trainer = pl.Trainer(**config["trainer"], logger=logger)

    if test_or_validation == "test":
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
