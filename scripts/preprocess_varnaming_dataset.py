import argparse
import gzip
import json
import math
import pathlib
from multiprocessing import Pool

import ijson

from data_processing.vocabulary.great_vocabulary import GreatVocabulary
from data_processing.vocabulary.spm_vocabulary import SPMVocabulary


# inplace
def change_dict(original_dict, vocabulary, max_token_length):
    ret_dict = original_dict.copy()
    ret_dict["ContextGraph"]["NodeLabels"] = {
        k: vocabulary.encode(v)[:max_token_length]
        for (k, v) in ret_dict["ContextGraph"]["NodeLabels"].items()
    }
    ret_dict["name"] = vocabulary.encode(ret_dict["name"])[:max_token_length]
    ret_dict["types"] = list(
        map(lambda t: t[:max_token_length], vocabulary.encode(ret_dict["types"]))
    )
    return ret_dict


def preprocess(files, vocabulary, max_token_length, max_node_count):
    file_from, file_to = files
    print(f"{file_from} -> {file_to}")
    gz_from = gzip.open(file_from, "rb")
    items_from = ijson.items(gz_from, "item")

    items_to = list(
        map(
            lambda x: change_dict(x, vocabulary, max_token_length),
            filter(
                lambda x: len(x["ContextGraph"]["NodeLabels"]) <= max_node_count
                if max_node_count != -1
                else True,
                items_from,
            ),
        )
    )
    json_to = json.dumps(items_to)
    with gzip.open(file_to, "wt") as gz_to:
        gz_to.write(json_to)
    gz_from.close()


def launch_preprocess(args, files, num):
    if args.vocabulary_type == "spm":
        vocabulary = SPMVocabulary(args.vocabulary_path)
    elif args.vocabulary_type == "great":
        vocabulary = GreatVocabulary(args.vocabulary_path)
    else:
        return

    max_token_length = args.max_token_length
    max_node_count = args.max_node_count

    per_worker = int(math.ceil(len(files) / args.num_workers))
    files_start = num * per_worker
    files_end = min(len(files), files_start + per_worker)
    files_slice = files[files_start:files_end].copy()

    for file in files_slice:
        preprocess(file, vocabulary, max_token_length, max_node_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dataset_path",
        help="Path to root dir of dataset, " "i.e. dir with train/ validation/ test/",
    )
    parser.add_argument(
        "output_dataset_path",
        help="Path to dir of preprocessed dataset where to put the "
        "preprocessed .json.gz files (other files are not copied!)",
    )

    parser.add_argument("vocabulary_type", help="Same as in config_varnaming.yaml")
    parser.add_argument("vocabulary_path", help="Same as in config_varnaming.yaml")
    parser.add_argument(
        "max_token_length", type=int, help="Same as in config_varnaming.yaml"
    )

    parser.add_argument("num_workers", type=int)

    parser.add_argument(
        "max_node_count", type=float, help="Same as in config_varnaming.yaml"
    )

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dataset_path)
    output_path = pathlib.Path(args.output_dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    if output_path.exists():
        raise ValueError(f"{output_path} already exists")

    if args.vocabulary_type not in ["spm", "great"]:
        raise ValueError(f"Unknown vocabulary type: {args.vocabulary_type}")

    files_to_preprocess = []
    for file in input_path.rglob("*.json.gz"):
        out_file = output_path / file.relative_to(input_path)
        if file.is_file():
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if file.suffixes[-2:] == [".json", ".gz"]:
                files_to_preprocess.append((file, out_file))

    with Pool(args.num_workers) as p:
        p.starmap(
            launch_preprocess,
            [(args, files_to_preprocess, i) for i in range(args.num_workers)],
        )


if __name__ == "__main__":
    main()
