import os
import sys
import zipfile

import gdown
import argparse

import yaml


def download_from_google_drive(root, link):
    os.makedirs(root)
    java_small_zip = gdown.download(
        link, root, resume=True, fuzzy=True
    )
    with zipfile.ZipFile(java_small_zip, "r") as zip_ref:
        zip_ref.extractall(root)
    os.remove(java_small_zip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root',
                        help="Path to directory where to dataset will be downloaded and unpacked",
                        required='--use-config' in sys.argv)
    parser.add_argument('--link',
                        help="Link to google drive file with dataset",
                        required='--use-config' in sys.argv)
    parser.add_argument('--use-config',
                        type=str,
                        help='Use setting from provided yaml config. '
                             'More precisely, takes data.root and data.link from config. '
                             'However, they will be overwritten by root or link options above')

    args = parser.parse_args()
    if args.use_config:
        with open("config_varnaming.yaml") as f:
            config = yaml.safe_load(f)
        root = config['data']['root']
        link = config['data']['link']
    if hasattr(args, 'root'):
        root = args.root
    if hasattr(args, 'link'):
        link = args.link

    download_from_google_drive(**vars(args))
