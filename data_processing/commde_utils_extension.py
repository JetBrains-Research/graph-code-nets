import os
from commode_utils.filesystem import get_lines_offsets, count_lines_in_file


def get_files_offsets(data_dir: str) -> list:
    files_offsets = list()
    for file_name in os.listdir(data_dir):
        files_offsets.append(get_lines_offsets(os.path.join(data_dir, file_name)))
    return files_offsets


def get_files_count_lines(data_dir: str) -> list:
    files_count_lines = list()
    cumulative_sum = 0
    for file_name in os.listdir(data_dir):
        cumulative_sum += count_lines_in_file(os.path.join(data_dir, file_name))
        files_count_lines.append(cumulative_sum)
    return files_count_lines


def get_file_index(files_count_lines: list, index: int) -> int:
    l, r = int(0), int(files_count_lines[-1])
    while r - l > 1:
        m = int((l + r) / 2)
        if files_count_lines[m] <= index:
            l = m
        else:
            r = m
    return l
