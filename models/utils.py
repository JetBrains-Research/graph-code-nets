from typing import List, Union

import torch


def generate_padding_mask(src_or_tgt, pad_id, device):
    return (src_or_tgt == pad_id).to(device)


def remove_special_symbols(out: torch.Tensor, spec_symbols) -> list[list[list[int]]]:
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
