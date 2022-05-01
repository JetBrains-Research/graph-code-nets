import torch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_padding_mask(src_or_tgt):
    return (
        torch.zeros_like(src_or_tgt)
        .masked_fill(src_or_tgt == 0, float("-inf"))
        .masked_fill(src_or_tgt != 0, float(0.0))
    )
