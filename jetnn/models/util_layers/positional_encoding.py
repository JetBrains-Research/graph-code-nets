import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_token_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_token_length).reshape(max_token_length, 1)
        pos_embedding = torch.zeros((max_token_length, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, : token_embedding.size(1), :])  # type: ignore
