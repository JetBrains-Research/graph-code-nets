import numpy as np
import torch


# Based on https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
def positional_encoding(
    dim: int, sentence_length: int, dtype=torch.float32
) -> torch.tensor:
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


def join_dicts(d1, d2) -> dict:
    return {**d1, **d2}


def prefix_sum(arr) -> list:
    res = [0]
    for a in arr:
        res.append(res[-1] + a)
    return res


def sparse_categorical_accuracy(y_true, y_pred) -> torch.tensor:
    return torch.eq(y_true, torch.argmax(y_pred, -1)).long()


def sparse_softmax_cross_entropy_with_logits(
    error_locations, loc_predictions
) -> torch.float32:
    loss_input = torch.log_softmax(loc_predictions, 1)
    loss_input = loss_input.type(torch.FloatTensor)
    labels = error_locations.type(torch.LongTensor)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    return loss(loss_input, labels)
