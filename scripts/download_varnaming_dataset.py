import os
import sys
import zipfile

import gdown
import argparse

import yaml


def download_from_google_drive(root, link):
    os.makedirs(root)
    java_small_zip = gdown.download(link, root, resume=True, fuzzy=True)
    with zipfile.ZipFile(java_small_zip, "r") as zip_ref:
        zip_ref.extractall(root)
    os.remove(java_small_zip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        help="Path to directory where to dataset will be downloaded and unpacked",
        required="--use-config" not in sys.argv,
    )
    parser.add_argument(
        "--link",
        help="Link to google drive file with dataset",
        required="--use-config" not in sys.argv,
    )
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Use setting from config_varnaming.yaml. "
        "However, they will be overwritten by root or link options above",
    )

    args = parser.parse_args()
    root = args.root
    link = args.link
    if args.use_config:
        with open("config_varnaming.yaml") as f:
            config = yaml.safe_load(f)
        if root is None:
            root = config["data"]["root"]
        if link is None:
            link = config["data"]["link"]

    download_from_google_drive(root, link)
