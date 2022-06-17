import math
import random

import numpy as np
import torch
from torch import nn


def generate_padding_mask(src_or_tgt, pad_id, device):
    return (src_or_tgt == pad_id).to(device)


def remove_special_symbols(l: list[int], spec_symbols) -> list[int]:
    return list(filter(lambda x: x not in spec_symbols, l))


def remove_special_symbols_t(out: torch.Tensor, spec_symbols) -> list[list[list[int]]]:
    """
    out shape: (batch_size, top_k, L)
    return: list of batch_size lists, whose length is <= L
    """

    ret = []
    spec_symbols_set = set(spec_symbols)
    for b_i in range(out.size(0)):
        ret_part = []
        for k in range(out.size(1)):
            filtered = list(
                filter(lambda x: x not in spec_symbols_set, out[b_i][k].tolist())
            )
            ret_part.append(filtered)
        ret.append(ret_part)
    return ret


def join_dicts(d1: dict, d2: dict) -> dict:
    return {**d1, **d2}


def positional_encoding(dim, sentence_length, dtype=torch.float32):
    encoded_vec = np.array(
        [
            pos / np.power(10000, 2 * i / dim)
            for pos in range(sentence_length)
            for i in range(dim)
        ]
    )
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def sparse_categorical_accuracy(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    return torch.eq(y_true, torch.argmax(y_pred, -1)).long()


def sparse_softmax_cross_entropy_with_logits(
    error_locations: torch.Tensor, loc_predictions: torch.Tensor
) -> torch.Tensor:
    loss_input = torch.log_softmax(loc_predictions, 1)
    loss_input = loss_input.type(torch.float)
    labels = error_locations.type(torch.long)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    return loss(loss_input, labels)


def fix_seed(seed, deterministic=False):
    # https://pytorch.org/docs/stable/notes/randomness.html

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, **kwargs):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, **kwargs)
        self.embedding_dim = embedding_dim

    def forward(self, tokens: Tensor) -> Tensor:  # type: ignore
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)
