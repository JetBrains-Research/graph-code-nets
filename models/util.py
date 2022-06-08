import torch
import numpy as np


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
