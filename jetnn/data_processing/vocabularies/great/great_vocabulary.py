from collections import defaultdict

from jetnn.data_processing.vocabularies.vocabulary import Vocabulary


class GreatVocabulary(Vocabulary):
    def __init__(self, vocab_path):
        self.bpe_lookup_dict = None
        self.bpe_cache = None
        self.vocab_dim = None
        self.w2i = None
        self.i2w = None
        self.vocab_path = vocab_path
        self.pad = "<PAD>"
        self.eow = "#"  # end of word
        self.load_vocab()

    def __len__(self):
        return self.vocab_dim

    def encode(self, token: str) -> list[int]:
        return self.translate(token)

    def decode(self, encoded: list[int]) -> str:
        return "".join(map(lambda i: self.i2w[i], encoded))

    def pad_id(self) -> int:
        return self.lookup(self.pad)

    def bos_id(self) -> int:
        return -1

    def eos_id(self) -> int:
        return -1

    def unk_id(self) -> int:
        return -1

    def load_vocab(self):
        with open(self.vocab_path, encoding="utf-8") as f:
            subtokens = [l.rstrip() for l in f]
        self.i2w = {ix + 1: w for ix, w in enumerate(subtokens)}
        self.i2w[0] = self.pad
        self.w2i = {w: ix for ix, w in self.i2w.items()}
        self.vocab_dim = len(self.i2w)

        self.bpe_cache = {}
        self.bpe_lookup_dict = defaultdict(set)
        for token in self.w2i.keys():
            self.bpe_lookup_dict[token[:2]].add(token)

    def translate(self, token, is_subtokenized=False):
        return (
            self.lookup(token)
            if is_subtokenized
            else [self.lookup(t) for t in self.tokenize(token)]
        )

    def lookup(self, token):
        return self.w2i[token] if token in self.w2i else self.w2i[self.pad]

    def tokenize(self, token):
        token += self.eow  # Add terminal symbol first
        tokens = []
        ix = 0
        if token in self.bpe_cache:
            return self.bpe_cache[token]
        while ix < len(token):
            if ix == len(token) - 2:
                tokens.append(token[ix:])
                break
            else:
                candidates = self.bpe_lookup_dict.get(token[ix : ix + 2], [])
                if not candidates:
                    top_candidate = token[ix]
                else:
                    candidates = [
                        t
                        for t in candidates
                        if t == token[ix : ix + len(t)]
                        and not len(token) == ix + len(t) + 1
                    ]
                    if not candidates:
                        top_candidate = token[ix]
                    else:
                        top_candidate = max(candidates, key=lambda e: len(e))
                tokens.append(top_candidate)
                ix += len(top_candidate)
        self.bpe_cache[token] = tokens
        return tokens
