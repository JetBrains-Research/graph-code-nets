import math
from typing import Optional

import pytorch_lightning as pl
from torch import nn, Tensor
from torch.nn import TransformerDecoderLayer, TransformerDecoder

from models.util_layers.positional_encoding import PositionalEncoding


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
        tgt: Tensor,  # shape: [batch size, src_seq_length]
        memory: Tensor,  # shape: [batch size, source_size, d_model], e.g. source_size is equal to 1 in VarNaming
        tgt_mask: Optional[Tensor] = None,  # shape: [src_seq_length, src_seq_length]
        memory_mask: Optional[Tensor] = None,  # shape: [source_size, src_seq_length]
        tgt_key_padding_mask: Optional[
            Tensor
        ] = None,  # shape: [batch size, src_seq_length]
        memory_key_padding_mask: Optional[
            Tensor
        ] = None,  # shape: [batch_size, source_size]
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
