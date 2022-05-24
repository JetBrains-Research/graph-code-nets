import os
import zipfile

import gdown
import argparse


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
    parser.add_argument('root', help="Path to directory where to dataset will be downloaded and unpacked")
    parser.add_argument('link', help="Link to google drive file with dataset")
    args = parser.parse_args()
    download_from_google_drive(**vars(args))
