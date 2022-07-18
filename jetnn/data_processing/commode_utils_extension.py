import os
import random

from commode_utils.filesystem import get_lines_offsets, count_lines_in_file
from bisect import bisect_left


def get_files_offsets(data_dir: str, debug: bool) -> list:
    files_offsets = list()
    files = os.listdir(data_dir)
    if not debug:
        random.shuffle(files)
    for file_name in os.listdir(data_dir):
        files_offsets.append(get_lines_offsets(os.path.join(data_dir, file_name)))
    return files_offsets


def get_files_count_lines(data_dir: str) -> list:
    files_pref_sum_lines = list()
    cumulative_sum = 0
    for file_name in os.listdir(data_dir):
        cumulative_sum += count_lines_in_file(os.path.join(data_dir, file_name))
        files_pref_sum_lines.append(cumulative_sum)
    return files_pref_sum_lines


def get_file_index(files_count_lines: list, index: int) -> tuple[int, int]:
    left_bound = bisect_left(files_count_lines, index)
    return left_bound, index - files_count_lines[left_bound]
