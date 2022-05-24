import heapq
import itertools
from dataclasses import dataclass, field
from queue import PriorityQueue

import pytorch_lightning as pl
import torch
import torch_scatter
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import Sequential
from torchvision.transforms import Lambda
import torch.nn.functional as F

from data_processing.vocabulary.vocabulary import Vocabulary
from models.gcn_encoder import GCNEncoder
from models.transformer_decoder import GraphTransformerDecoder
from models.utils import generate_square_subsequent_mask, generate_padding_mask

from numpy import inf


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
            raise ValueError(f"Unknown decoder type: {self.config['model']['decoder']}")

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id())

    def forward(self, batch: Batch) -> Tensor:  # type: ignore
        varname_batch: torch.Tensor = self.encoder(batch)

        if self.config["model"]["decoder"] == "transformer_decoder":
            target_batch = batch.name

            target_mask = generate_square_subsequent_mask(self.max_token_length)

            # TODO: investigate if this mask has any effect
            target_padding_mask = generate_padding_mask(
                target_batch, self.vocabulary.pad_id()
            )

            predicted = self.decoder(
                target_batch,  # shape: [batch size, src_emb_size]
                varname_batch,  # shape: [batch size, 1, d_model]
                tgt_mask=target_mask,  # shape: [src_emb_size, src_emb_size]
                tgt_key_padding_mask=target_padding_mask,  # shape: [batch size, src_emb_size]
            )  # shape: [batch size, src_emb_size, target vocabulary dim]

            return predicted
        else:
            raise ValueError(f"Unknown decoder type: {self.config['model']['decoder']}")

    @torch.no_grad()
    def generate(
        self, batch: Batch, method="beam_search", bandwidth=10, top_k=1, max_steps=5000
    ):  # batch without `name` attribute
        varname_batch: torch.Tensor = self.encoder(batch)

        if self.config["model"]["decoder"] == "transformer_decoder":
            if method == "greedy":
                generated = []
                for b_i in range(varname_batch.size(0)):
                    varname_batch_part = varname_batch[
                        b_i : b_i + 1, :
                    ]  # keep batch dimension

                    current = torch.ones((1, 1)).fill_(self.vocabulary.bos_id())

                    for i in range(
                        self.max_token_length - 2
                    ):  # bos + variable name + eos
                        target_mask = generate_square_subsequent_mask(current.size(1))

                        predicted = self.decoder(
                            current, varname_batch_part, tgt_mask=target_mask
                        )
                        _, next_word_batch = torch.max(predicted[:, -1, :], dim=1)
                        next_word = next_word_batch.item()

                        if next_word == self.vocabulary.eos_id():
                            break

                        current = torch.cat(
                            [
                                current,
                                torch.ones(1, 1).type_as(current.data).fill_(next_word),
                            ],
                            dim=1,
                        )

                    current = torch.cat(
                        [current, torch.ones((1, 1)).fill_(self.vocabulary.eos_id())],
                        dim=1,
                    )
                    padded = F.pad(
                        current, (0, self.max_token_length - current.size(1))
                    )
                    generated.append(padded)
                generated_batch = torch.cat(
                    generated, dim=0
                ).int()  # shape: (batch_size, max_token_length)
                return generated_batch
            elif method == "beam_search":

                @dataclass(order=True)
                class BeamNode:
                    score: float  # negative log probability (0 is the best, inf is the worst)
                    state: torch.Tensor = field(compare=False)  # shape: (1, length)

                generated = []
                for b_i in range(varname_batch.size(0)):
                    generated_part = []
                    varname_batch_part = varname_batch[
                        b_i : b_i + 1, :
                    ]  # keep batch dimension

                    current_state = torch.ones((1, 1)).fill_(self.vocabulary.bos_id())

                    steps = 0
                    pq: list[BeamNode] = []
                    heapq.heappush(pq, BeamNode(0.0, current_state))

                    while True:
                        steps += 1
                        node = heapq.heappop(pq)
                        current_score = node.score
                        current_state = node.state

                        max_len_reached = (
                            current_state.size(1) >= self.max_token_length - 1
                        )
                        eos_reached = (
                            current_state[:, -1].item() == self.vocabulary.eos_id()
                        )
                        max_steps_reached = steps > max_steps

                        if max_len_reached or eos_reached or max_steps_reached:

                            # If not eos_reached, then we need to add eos to the end
                            if not eos_reached:
                                current_state = torch.cat(
                                    [
                                        current_state,
                                        torch.ones((1, 1)).fill_(
                                            self.vocabulary.eos_id()
                                        ),
                                    ],
                                    dim=1,
                                )

                            # If not max_len_reached, then we need to add padding
                            if not max_len_reached:
                                current_state = F.pad(
                                    current_state,
                                    (0, self.max_token_length - current_state.size(1)),
                                )

                            generated_part.append(current_state)

                            if len(generated_part) >= top_k:
                                break
                            else:
                                continue

                        target_mask = generate_square_subsequent_mask(
                            current_state.size(1)
                        )

                        # shape: (1, length, target_vocabulary_size)
                        predicted = self.decoder(
                            current_state, varname_batch_part, tgt_mask=target_mask
                        )
                        neg_log_prob_predicted = -F.log_softmax(predicted, dim=2)
                        top_scores, top_indices = torch.topk(
                            neg_log_prob_predicted[:, -1, :], bandwidth, dim=1
                        )

                        for i in range(top_scores.size(1)):
                            new_score = current_score + top_scores[:, i].item()
                            new_state = torch.cat(
                                [current_state, top_indices[:, i : i + 1]], dim=1
                            )
                            heapq.heappush(
                                pq, BeamNode(score=new_score, state=new_state)
                            )
                    generated_batch_part = torch.cat(
                        generated_part, dim=0
                    ).int()  # shape: (top_k, max_token_length)
                    generated.append(generated_batch_part)
                generated_batch = torch.stack(
                    generated, dim=0
                )  # shape: (batch_size, top_k, max_token_length)
                return generated_batch
            else:
                raise ValueError(f"Unknown method: {method}")

    def _shared_step(self, batch: Batch, batch_idx: int, step: str) -> STEP_OUTPUT:
        predicted = self(batch)
        loss = self.loss_fn(
            predicted.reshape(-1, predicted.shape[-1]), batch.name.long().reshape(-1)
        )
        self.log(
            f"{step}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self._shared_step(batch, batch_idx, "validation")

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self._shared_step(batch, batch_idx, "test")

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
