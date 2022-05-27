import pytorch_lightning as pl

import yaml

from data_processing.graph_var_miner_dataloader import GraphVarMinerModule
from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.vocabulary.spm_vocabulary import SPMVocabulary
from models.varnaming import VarNamingModel


def main():
    with open("config_varnaming.yaml") as f:
        config = yaml.safe_load(f)

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

    datamodule = GraphVarMinerModule(
        config,
        vocabulary,
    )
    model = VarNamingModel(config, vocabulary)

    trainer = pl.Trainer(**config["train"]["trainer"])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
