import itertools

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import Batch

from data_processing.vocabulary.vocabulary import Vocabulary
from models.gcn_encoder import GCNEncoder
from models.transformer_decoder import GraphTransformerDecoder
from models.utils import generate_square_subsequent_mask, generate_padding_mask


class VarNamingModel(pl.LightningModule):
    def __init__(self, config: dict, vocabulary: Vocabulary) -> None:
        super().__init__()

        self.config = config
        self.vocabulary = vocabulary
        self.max_token_length = self.config["vocabulary"]["max_token_length"]

        encoder_config = self.config["model"][self.config["model"]["encoder"]]
        if self.config["model"]["encoder"] == "gcn":
            self.encoder = GCNEncoder(
                **encoder_config,
                in_channels=self.max_token_length,
            )  # torch_geometric.data.Data -> torch.Tensor [num_nodes, d_model]
        else:
            raise ValueError(f"Unknown encoder type: {self.config['model']['encoder']}")

        decoder_config = self.config["model"][self.config["model"]["decoder"]]
        if self.config["model"]["decoder"] == "transformer_decoder":
            self.decoder = GraphTransformerDecoder(
                **decoder_config,
                target_vocab_size=len(vocabulary),
                max_tokens_length=self.max_token_length,
            )
        else:
            raise ValueError(f"Unknown decoder type: {self.config['model']['encoder']}")

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id())

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

        # TODO: perhaps I should put some of this code into decoder, as this is very transformer-specific code
        # or either make some if-s like: if isinstance(self.decoder, GraphTransformerDecoder):
        target_batch = batch.name

        target_mask = generate_square_subsequent_mask(self.max_token_length)

        # TODO: investigate if this mask has any effect
        target_padding_mask = generate_padding_mask(target_batch)

        predicted = self.decoder(
            target_batch,  # shape: [batch size, src_emb_size]
            varname_batch,  # shape: [batch size, 1, d_model]
            tgt_mask=target_mask,  # shape: [src_emb_size, src_emb_size]
            tgt_key_padding_mask=target_padding_mask,  # shape: [batch size, src_emb_size]
        )  # shape: [batch size, src_emb_size, target vocabulary dim]

        return predicted

    def _shared_step(self, batch: Batch, batch_idx: int, step: str) -> STEP_OUTPUT:
        predicted = self(batch)
        loss = self.loss_fn(
            predicted.reshape(-1, predicted.shape[-1]), batch.name.long().reshape(-1)
        )
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        loss = self._shared_step(batch, batch_idx, "train")
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        pass

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        pass

    def configure_optimizers(self):
        # TODO add optimizer config
        optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.config["train"]["learning_rate"],
        )
        # TODO lr_scheduler?
        return {
            "optimizer": optimizer,
        }

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:  # type: ignore
    #     pass
