import gzip
import pathlib
import random
from itertools import chain
from typing import Iterable, Union

import ijson
import sentencepiece as spm

from data_processing.vocabulary.vocabulary import Vocabulary


class SPMVocabularyTrainer:
    def __init__(
        self,
        root: str,
        vocab_size: int,
        model_type: str,
        model_prefix: str = "spm",
        fraction_prob=0.1,
        seed=1337,
        num_threads=4,
        **kwargs
    ):
        self._root = root
        self._vocab_size = vocab_size
        self._model_type = model_type
        self._model_prefix = model_prefix
        self._fraction_prob = (
            fraction_prob  # random sample only fraction_prob of all words
        )
        self._seed = seed
        self._num_threads = 4
        self._kwargs = kwargs

        _raw_data_path = pathlib.Path(self._root)
        if not _raw_data_path.exists():
            raise FileNotFoundError(self._root)

        self._data_files: list[pathlib.Path] = [
            f for f in _raw_data_path.iterdir() if f.is_file()
        ]

    def _words_from_file(self, filename) -> Iterable[str]:
        f = gzip.open(str(filename), "rb")
        items = ijson.items(f, "item.ContextGraph.NodeLabels")
        return filter(
            lambda _: random.random() < self._fraction_prob,
            chain.from_iterable(map(lambda p: iter(p.values()), items)),
        )

    def _word_iterator(self):
        return chain.from_iterable(map(self._words_from_file, self._data_files))

    def train(self):
        random.seed(self._seed)
        words_iterator = self._word_iterator()
        spm.SentencePieceTrainer.Train(
            sentence_iterator=words_iterator,
            vocab_size=self._vocab_size,
            model_type=self._model_type,
            model_prefix=self._model_prefix,
            add_dummy_prefix=False,  # in GraphVarMiner dataset no whitespaces exist,
            # and words should be generated in camel case, so dummy whitespace is redundant
            num_threads=self._num_threads,
            **self._kwargs
        )


class SPMVocabulary(Vocabulary):
    def __init__(self, model_file: str):
        self._model_file = model_file
        self._model = spm.SentencePieceProcessor()
        self._model.Init(add_bos=True, add_eos=True, model_file=self._model_file)

    def __len__(self) -> int:
        return len(self._model)

    def encode(self, token: str) -> list[int]:
        return self._model.Encode(token)

    def decode(self, encoded: Union[int, list[int], list[list[int]]]) -> str:
        return self._model.Decode(encoded)

    def pad_id(self) -> int:
        return self._model.pad_id()

    def bos_id(self) -> int:
        return self._model.bos_id()

    def eos_id(self) -> int:
        return self._model.eos_id()

    def unk_id(self) -> int:
        return self._model.unk_id()
