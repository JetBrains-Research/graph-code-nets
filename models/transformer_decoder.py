import math
from typing import Optional, Callable

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import TransformerDecoderLayer, TransformerDecoder


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


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:  # type: ignore
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class GraphTransformerDecoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        target_vocab_size: int,
        max_tokens_length: int,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        self.tgt_tok_emb = TokenEmbedding(target_vocab_size, d_model)
        self.generator = nn.Linear(d_model, target_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.generator(
            self.decoder(
                tgt=self.positional_encoding(self.tgt_tok_emb(tgt)),
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,  # TODO no changes caused by this mask? investigate
                memory_key_padding_mask=memory_key_padding_mask,
            )
        )
