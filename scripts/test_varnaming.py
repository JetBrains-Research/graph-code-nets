import sys

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger


def main():
    if len(sys.argv) < 2:
        raise ValueError("Expecting ckpt_path")

    ckpt_path = sys.argv[1]

    config_path = (
        "config_varnaming.yaml"
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.save(config_path, policy="now")

    logger = WandbLogger(**config["logger"])

    trainer = pl.Trainer(logger=logger)

    trainer.test(ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
