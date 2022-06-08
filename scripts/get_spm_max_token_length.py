import gzip
import pathlib
import sys
import time
from datetime import datetime
from itertools import chain
from typing import Iterable

from collections import defaultdict

import ijson
import numpy as np

from data_processing.vocabulary.spm_vocabulary import SPMVocabulary

counter = 0


def words_from_file(filename) -> Iterable[str]:
    global counter
    print(counter, filename)
    counter += 1
    f = gzip.open(str(filename), "rb")
    items = ijson.items(f, "item.ContextGraph.NodeLabels")
    return chain.from_iterable(map(lambda p: iter(p.values()), items))


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

    it = word_iterator(data_files)

    enc_lens = defaultdict(int)
    mx = 0
    for word in it:
        enc = vocab.encode(word)
        enc_lens[len(enc)] += 1
        if len(enc) > mx:
            mx = len(enc)
            print(f"{datetime.fromtimestamp(time.time())}: New maximum {mx} on {word}")
    enc_lens_l = [0] * (max(enc_lens) + 1)
    for k, v in enc_lens.items():
        enc_lens_l[k] = v

    print("Maximum length is ", mx)
    print("Distribution is ", enc_lens_l)

    dists = np.array(enc_lens_l, dtype=float)
    dists /= np.sum(dists)
    dists = np.cumsum(dists)
    percs = [0.95, 0.99, 0.999, 0.9999, 1.0]
    percs_r = np.zeros_like(percs, dtype=bool)
    percs_v = np.zeros_like(percs, dtype=int)
    for n, p in enumerate(dists):
        for i in range(len(percs)):
            if p > percs[i] and not percs_r[i]:
                percs_r[i] = True
                percs_v[i] = n
    percs_v[-1] = mx

    print(f"Dataset {root}")
    print(f"Model {model}")
    print(f"Percentiles: {list(percs)}")
    print(f"Max token length: {list(percs_v)}")


if __name__ == "__main__":
    main()
