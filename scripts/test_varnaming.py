import sys

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger

from data_processing.graph_var_miner.graph_var_miner_dataloader import (
    GraphVarMinerModule,
)
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.vocabulary.spm_vocabulary import SPMVocabulary
from models.varnaming import VarNamingModel


def main():
    if len(sys.argv) < 4:
        raise ValueError(
            "Expecting test/validate, dataset type (train/validation/test), ckpt_path"
        )

    test_or_validation = sys.argv[1]
    if test_or_validation not in ["test", "validation"]:
        raise ValueError(f"Unexpected test_or_validate: {test_or_validation}")

    dataset_type = sys.argv[2]

    ckpt_path = sys.argv[3]

    config_path = "config_varnaming.yaml"

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
    if dataset_type == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    elif dataset_type == "validation":
        datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()
    elif dataset_type == "train":
        datamodule.setup("fit")
        dataloader = datamodule.train_dataloader()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    model = VarNamingModel.load_from_checkpoint(
        checkpoint_path=ckpt_path, config=config, vocabulary=vocabulary
    )

    trainer = pl.Trainer(**config["trainer"], logger=logger)

    if test_or_validation == "test":
        trainer.test(model=model, dataloaders=dataloader, ckpt_path=ckpt_path)
    else:
        trainer.validate(model=model, dataloaders=dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
