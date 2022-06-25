from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, tensor

from data_processing.vocabulary.vocabulary import Vocabulary
from models.utils import TokenEmbedding


class DecoderGRU(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        target_vocab_size: int,
        max_tokens_length: int,
        vocabulary: Vocabulary,
        embedding: TokenEmbedding,
    ) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = torch.nn.Linear(
            hidden_size * num_layers, target_vocab_size, bias=False
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_tokens_length = max_tokens_length
        self.num_layers = num_layers
        self.target_vocab_size = target_vocab_size
        self.vocabulary = vocabulary
        self.embedding = embedding

    def forward(
        self,
        enc: Tensor,  # shape: [batch size, 1, d_model]
        tgt: Tensor = None,  # already embedded, shape: [batch size, src_seq_length, d_model == embedding_dim]
    ) -> Any:
        # shape: [num_layers, batch size, d_model]
        h = enc.transpose(0, 1).repeat(self.num_layers, 1, 1)
        # shape: [batch_size, max output sequence length, size of target vocabulary]
        output = torch.full(
            (enc.size(0), self.max_tokens_length, self.target_vocab_size),
            self.vocabulary.pad_id(),
            dtype=torch.int,
            device=self.device,
        )
        output[:, 0, self.vocabulary.bos_id()] = 1
        output[:, -1, self.vocabulary.eos_id()] = 1
        # shape: [batch_size, 1]
        current_input = torch.full(
            (enc.size(0), 1), self.vocabulary.bos_id(), dtype=torch.int
        )
        # shape: [batch_size, 1, embedding size == d_model == input size]
        current_input = self.embedding(current_input)
        for i in range(1, self.max_tokens_length - 1):
            # shape: [batch_size, 1, hidden_size*num_layers], [num_layers, batch_size, hidden_size]
            current_output, h = self.gru(current_input, h)
            # shape: [batch_size, 1, vocab_size]
            proj = self.projection(current_output)
            output[:, i, :] = proj
            if tgt is not None:
                current_input = tgt[:, i, :]
            else:
                # shape: [batch_size, 1]
                current_input = proj.argmax(dim=-1)
                # shape: [batch_size, 1, embedding_size == d_model == input size]
                current_input = self.embedding(current_input)
            all_eos = (
                (output[:, :-1].argmax(dim=-1) == self.vocabulary.eos_id())
                .any(dim=1)
                .all(dim=0)
                .item()
            )
            if all_eos:
                break
        return output
