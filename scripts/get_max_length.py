import gzip
import pathlib
import random
import sys
from itertools import chain
from typing import Iterable

import ijson

from data_processing.vocabulary.spm_vocabulary import (
    SPMVocabulary,
)

fraction_prob = 0.1
seed = 1337


def words_from_file(filename) -> Iterable[str]:
    f = gzip.open(str(filename), "rb")
    items = ijson.items(f, "item.ContextGraph.NodeLabels")
    return filter(
        lambda _: random.random() < fraction_prob,
        chain.from_iterable(
            map(lambda p:
                iter(p.values()),
                items)
        )
    )


def word_iterator(data_files):
    return chain.from_iterable(map(words_from_file, data_files))


def main():
    root = sys.argv[1]
    model = sys.argv[2]

    vocab = SPMVocabulary(model_file=model)

    _raw_data_path = pathlib.Path(root)
    if not _raw_data_path.exists():
        raise FileNotFoundError(root)

    data_files: list[pathlib.Path] = [
        f for f in _raw_data_path.iterdir() if f.is_file()
    ]
    random.seed(seed)

    it = word_iterator(data_files)

    mx = 0
    for word in it:
        enc = vocab.encode(word)
        if len(enc) > mx:
            mx = len(enc)
            print(f'New maximum {mx} on {word}')

    print('Maximum length is ', mx)


if __name__ == "__main__":
    main()