import argparse
import json
import os
import yaml

filename_pref = "processed_"
dirs_to_process = ["train", "dev", "eval"]

ap = argparse.ArgumentParser()
ap.add_argument("config_path")
args = ap.parse_args()

config_path = args.config_path
config = yaml.safe_load(open(config_path))


def process_data(config: dict):
    raw_data_path = config["paths"]["raw_data"]
    for process_dir in dirs_to_process:
        path_to_process_dir = os.path.join(raw_data_path, process_dir)
        path_to_create_dir = os.path.join(raw_data_path, filename_pref + process_dir)
        os.makedirs(path_to_create_dir, exist_ok=True)
        if path_to_process_dir not in os.listdir(raw_data_path):
            for filename in os.listdir(path_to_process_dir):
                path_to_file_in = os.path.join(path_to_process_dir, filename)
                path_to_file_out = os.path.join(
                    path_to_create_dir, filename_pref + filename
                )
                with open(path_to_file_in, "r") as file_in, open(
                    path_to_file_out + filename, "w"
                ) as file_out:
                    for line in file_in.readlines():
                        if (
                            len(json.loads(line)["source_tokens"])
                            <= config["data"]["drop_sequence_length"]
                        ):
                            file_out.write(line)
