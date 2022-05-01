from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import Batch

from data_processing.vocabulary import Vocabulary
from models.gcn_encoder import GCNEncoder
from models.transformer_decoder import GraphTransformerDecoder
from models.utils import generate_square_subsequent_mask, generate_padding_mask


class VarNamingModel(pl.LightningModule):
    def __init__(self, vocabulary: Vocabulary) -> None:
        super().__init__()

        self.src_emb_size = 10
        self.encoder = GCNEncoder(
            self.src_emb_size, 32
        )  # torch_geometric.data.Data -> torch.Tensor
        self.decoder = GraphTransformerDecoder(
            6, 32, 8, vocabulary.vocab_dim
        )  # torch.Tensor -> torch.Tensor

    def forward(self, batch: Batch) -> Tensor:  # type: ignore
        enc = self.encoder(batch)
        graph_sizes = []
        for i in range(batch.num_graphs):
            graph_sizes.append(batch[i].num_nodes)

        masked_enc = enc * batch.marked_tokens.unsqueeze(-1)
        separated_enc = torch.split(masked_enc, graph_sizes, dim=0)
        varname_batch = torch.cat(
            list(map(lambda x: torch.mean(x, dim=0).unsqueeze(0), separated_enc)), dim=0
        ).unsqueeze(1)

        target_batch = batch.name

        target_mask = generate_square_subsequent_mask(self.src_emb_size)

        # TODO: investigate if this mask has any effect
        target_padding_mask = generate_padding_mask(target_batch)

        return self.decoder(
            target_batch,  # shape: [batch size, src_emb_size]
            varname_batch,  # shape: [batch size, 1, 32]
            tgt_mask=target_mask,  # shape: [src_emb_size, src_emb_size]
            tgt_key_padding_mask=target_padding_mask,
        )  # shape: [batch size, src_emb_size]

    # тут должно быть разделение полученного линейного слоя на графы по batch.batch, и далее выбор из мест с <var> по marked_tokens, берёшь среднее, и из полученного строишь батч,и вот его передаешь вместо self.encoder(batch) выше

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)
