import heapq
import itertools
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Transformer
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torchmetrics.functional import chrf_score

from data_processing.identifiersplitting import split_identifier_into_parts
from data_processing.vocabulary.vocabulary import Vocabulary
from models.gcn_encoder import GCNEncoder
from models.transformer_decoder import GraphTransformerDecoder
from models.utils import generate_padding_mask, remove_special_symbols


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

            target_mask = Transformer.generate_square_subsequent_mask(
                self.max_token_length
            ).to(self.device)

            # TODO: investigate if this mask has any effect
            target_padding_mask = generate_padding_mask(
                target_batch, self.vocabulary.pad_id(), device=self.device
            )

            predicted = self.decoder(
                target_batch,  # shape: [batch size, src_seq_length]
                varname_batch,  # shape: [batch size, 1, d_model]
                tgt_mask=target_mask,  # shape: [src_seq_length, src_seq_length]
                tgt_key_padding_mask=target_padding_mask,  # shape: [batch size, src_seq_length]
            )  # shape: [batch size, src_seq_length, target vocabulary dim]

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
                generated_batch = torch.ones(
                    (varname_batch.size(0), 1, self.max_token_length),
                    dtype=torch.int,
                    device=self.device,
                ).fill_(self.vocabulary.pad_id())
                for b_i in range(varname_batch.size(0)):
                    varname_batch_part = varname_batch[
                        b_i : b_i + 1, :
                    ]  # keep batch dimension

                    current = torch.ones((1, 1), device=self.device).fill_(
                        self.vocabulary.bos_id()
                    )

                    for i in range(
                        self.max_token_length - 2
                    ):  # bos + variable name + eos
                        target_mask = Transformer.generate_square_subsequent_mask(
                            current.size(1)
                        ).to(self.device)

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

                    generated_batch[b_i : b_i + 1, 0, : current.size(1)] = current
                    generated_batch[b_i, 0, current.size(1)] = self.vocabulary.eos_id()
                return generated_batch
            elif method == "beam_search":

                @dataclass(order=True)
                class BeamNode:
                    score: float  # negative log probability (0 is the best, inf is the worst)
                    state: torch.Tensor = field(compare=False)  # shape: (1, length)

                generated_batch = torch.ones(
                    (varname_batch.size(0), top_k, self.max_token_length),
                    dtype=torch.int,
                    device=self.device,
                ).fill_(self.vocabulary.pad_id())
                for b_i in range(varname_batch.size(0)):
                    generated_part_n = 0
                    varname_batch_part = varname_batch[
                        b_i : b_i + 1, :
                    ]  # keep batch dimension

                    current_state = torch.ones((1, 1), device=self.device).fill_(
                        self.vocabulary.bos_id()
                    )

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
                            # padding is added automatically (generated_batch is filled with pad_id)
                            generated_batch[
                                b_i : b_i + 1, generated_part_n, : current_state.size(1)
                            ] = current_state

                            # If not eos_reached, then we need to add eos to the end
                            if not eos_reached:
                                generated_batch[
                                    b_i, generated_part_n, current_state.size(1)
                                ] = self.vocabulary.eos_id()

                            generated_part_n += 1

                            if generated_part_n >= top_k:
                                break
                            else:
                                continue

                        target_mask = Transformer.generate_square_subsequent_mask(
                            current_state.size(1)
                        ).to(self.device)

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
                return generated_batch
            else:
                raise ValueError(f"Unknown method: {method}")

    def _shared_step(
        self, batch: Batch, batch_idx: int, step: str
    ) -> tuple[Tensor, Tensor]:
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
        self.log(
            f"{step}_items_processed",
            float(batch_idx * batch.num_graphs),
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return predicted, loss

    def _generate_step(self, batch: Batch, batch_idx: int, step: str):
        generation_config = self.config[step]["generation"]

        method = generation_config["method"]
        bandwidth = int(generation_config["bandwidth"])
        max_steps = int(generation_config["max_steps"])
        mrr_k = int(generation_config["mrr_k"])
        acc_k = int(generation_config["acc_k"])
        max_generate_k = max(mrr_k, acc_k)

        generated = self.generate(
            batch,
            method=method,
            bandwidth=bandwidth,
            top_k=max_generate_k,
            max_steps=max_steps,
        )  # (batch, top_k, dim)

        return generated

    # input_t: (batch, top_k, dim)
    def _log_metrics(self, batch: Batch, input_t: Tensor, batch_idx: int, step: str):
        generation_config = self.config[step]["generation"]

        mrr_k = int(generation_config["mrr_k"])
        acc_k = int(generation_config["acc_k"])

        target_t = (
            to_dense_batch(batch.name)[0].transpose(0, 1).int()
        ).int()  # (batch, 1, dim)  # 1 because there is only 1 name in sample

        chrf = torch.tensor(0.0, device=self.device)

        for input_, target_ in zip(input_t[:, 0], target_t[:, 0]):
            input_dec = self.vocabulary.decode(
                remove_special_symbols(
                    input_.tolist(),
                    [self.vocabulary.pad_id(), self.vocabulary.unk_id()],
                )
            )
            target_dec = self.vocabulary.decode(
                remove_special_symbols(
                    target_.tolist(),
                    [self.vocabulary.pad_id(), self.vocabulary.unk_id()],
                )
            )
            input_words = " ".join(split_identifier_into_parts(input_dec))
            target_words = " ".join(split_identifier_into_parts(target_dec))
            chrf += chrf_score([input_words], [target_words])
        chrf /= input_t.shape[0]

        eqs = input_t.eq(target_t)  # (batch, top_k, dim)
        exact_eqs = eqs.all(dim=2)  # (batch, top_k)
        acc_exact_1 = exact_eqs[0].float().mean()  # exact name
        acc_exact_k = exact_eqs[:acc_k].any(dim=1).float().mean()  # exact name

        ranks_mask = (
            exact_eqs[:, :mrr_k].any(dim=1).float()
        )  # (batch)  # if not found, then inv rank is 0
        arange = torch.arange(mrr_k, 0, -1, device=self.device).unsqueeze(
            0
        )  # (batch, mrr_k)
        ranks = torch.argmax(exact_eqs[:, :mrr_k].float() * arange, dim=1)  # (batch)
        mrr_exact_k = torch.mean(1 / (ranks + 1) * ranks_mask)

        self.log(
            "chrf",
            chrf,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"accuracy_exact_top{1}",
            acc_exact_1,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"accuracy_exact_top{acc_k}",
            acc_exact_k,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"mrr_exact_top{mrr_k}",
            mrr_exact_k,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        _, loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        predicted, loss = self._shared_step(batch, batch_idx, "validation")
        predicted_best = torch.argmax(predicted, dim=2).unsqueeze(1)
        self._log_metrics(batch, predicted_best, batch_idx, "validation")
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        _, loss = self._shared_step(batch, batch_idx, "test")
        generated = self._generate_step(batch, batch_idx, "test")
        self._log_metrics(batch, generated, batch_idx, "test")
        return loss

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
