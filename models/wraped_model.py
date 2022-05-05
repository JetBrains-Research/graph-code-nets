import numpy as np
import models.util as util
import torch
import torch.nn.functional as F
from models import two_pointer_fcn, encoder_gru, encoder_ggnn
import pytorch_lightning as pl
from models.util import (
    sparse_categorical_accuracy,
    sparse_softmax_cross_entropy_with_logits,
)


class VarMisuseLayer(pl.LightningModule):
    def __init__(self, config: dict, vocab_dim: int):
        super().__init__()
        self._model_config = config["model"]
        self._data_config = config["data"]
        self._training_config = config["training"]
        self._vocab_dim = vocab_dim
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._embedding = torch.nn.Embedding(
            self._vocab_dim, self._model_config["base"]["hidden_dim"]
        )

        base_config = self._model_config["base"]
        inner_model = self._model_config["configuration"]
        self._prediction = two_pointer_fcn.TwoPointerFCN(base_config)
        if inner_model == "rnn":
            self._model = encoder_gru.EncoderGRU(
                util.join_dicts(base_config, self._model_config["rnn"])
            )
        elif inner_model == "ggnn":
            self._model = encoder_ggnn.EncoderGGNN(
                util.join_dicts(base_config, self._model_config["ggnn"])
            )
        else:
            raise ValueError("Unknown model component provided:", inner_model)

    def forward(
        self, tokens: torch.tensor, token_mask: torch.tensor, edges: torch.tensor
    ) -> torch.tensor:
        original_shape = list(
            np.append(np.array(tokens.shape), self._model_config["base"]["hidden_dim"])
        )
        flat_tokens = tokens.type(torch.LongTensor).flatten().to(self._device)
        subtoken_embeddings = self._embedding(flat_tokens)
        subtoken_embeddings = torch.reshape(subtoken_embeddings, original_shape)
        subtoken_embeddings *= torch.unsqueeze(torch.clamp(tokens, 0, 1), -1).to(
            self._device
        )
        states = torch.mean(subtoken_embeddings, 2)
        if self._model_config["configuration"] == "rnn":
            predictions = torch.transpose(
                self._prediction(self._model(states)[0]), 1, 2
            )
            return predictions

        elif self._model_config["configuration"] == "ggnn":
            predictions = list()
            for i in range(len(tokens)):
                test_predictions = self._model(
                    states[i].float(),
                    torch.transpose(
                        torch.tensor([[e[1], e[2]] for e in edges if e[0] == i]), 0, 1
                    ).to(self._device),
                )
                predictions.append(test_predictions)
            predictions = torch.stack(predictions).to(self._device)
            predictions = torch.transpose(self._prediction(predictions), 1, 2)
            return predictions

    def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.float32:
        is_buggy, loc_predictions, target_probs = self._shared_train_eval_step(
            batch, batch_idx
        )
        error_loc = batch[2]

        ls = self.test_get_loss(is_buggy, loc_predictions, target_probs, error_loc)
        loss = sum(ls.values())
        self.log(
            "training_loss",
            ls,
            prog_bar=True,
            on_epoch=True,
            batch_size=self._data_config["batch_size"],
        )
        return loss

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.float32:
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.tensor, batch_idx: int) -> torch.float32:
        return self._shared_eval_step(batch, batch_idx, "test")

    def _shared_train_eval_step(self, batch: torch.tensor, batch_idx: int):
        tokens, edges, error_loc, repair_targets, repair_candidates = batch
        token_mask = torch.clamp(torch.sum(tokens, -1), 0, 1)
        pointer_preds = self(tokens, token_mask, edges)
        is_buggy, loc_predictions, target_probs = self._shared_loss_acs_calc(
            pointer_preds, token_mask, error_loc, repair_targets, repair_candidates
        )
        return is_buggy, loc_predictions, target_probs

    def _shared_eval_step(
        self, batch: torch.tensor, batch_idx: int, step: str
    ) -> torch.float32:
        is_buggy, loc_predictions, target_probs = self._shared_train_eval_step(
            batch, batch_idx
        )
        error_loc = batch[2]

        ls = self.test_get_loss(is_buggy, loc_predictions, target_probs, error_loc)
        acs = self.test_get_acs(is_buggy, loc_predictions, target_probs, error_loc)
        loss = sum(ls.values())
        self.log(
            step + "_loss",
            ls,
            prog_bar=True,
            on_epoch=True,
            batch_size=self._data_config["batch_size"],
        )
        self.log(
            step + "_acc",
            acs,
            prog_bar=True,
            on_epoch=True,
            batch_size=self._data_config["batch_size"],
        )
        return loss

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(
            self.parameters(), lr=self._training_config["learning_rate"]
        )

    def _shared_loss_acs_calc(
        self,
        predictions: torch.tensor,
        token_mask: torch.tensor,
        error_locations: torch.tensor,
        repair_targets: torch.tensor,
        repair_candidates: torch.tensor,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        seq_mask = token_mask.float()
        predictions += (1.0 - torch.unsqueeze(seq_mask, 1)) * torch.finfo(
            torch.float32
        ).min
        is_buggy = torch.clamp(error_locations, 0, 1).float()
        loc_predictions = predictions[:, 0]
        pointer_logits = predictions[:, 1]

        candidate_mask = np.zeros(pointer_logits.size())
        for e in repair_candidates:
            candidate_mask[e[0]][e[1]] = 1
        candidate_mask = torch.tensor(candidate_mask).to(self._device)

        pointer_logits += (1.0 - candidate_mask) * torch.finfo(torch.float32).min
        pointer_probs = F.softmax(pointer_logits, dim=-1)

        target_mask = np.zeros(pointer_probs.size())
        for e in repair_targets:
            target_mask[e[0]][e[1]] = 1
        target_mask = torch.tensor(target_mask).to(self._device)
        target_probs = torch.sum(target_mask * pointer_probs, -1)

        return is_buggy, loc_predictions, target_probs

    def test_get_loss(
        self,
        is_buggy: torch.tensor,
        loc_predictions: torch.tensor,
        target_probs: torch.tensor,
        error_locations: torch.tensor,
    ) -> dict[str, torch.tensor]:
        loc_loss = sparse_softmax_cross_entropy_with_logits(
            error_locations, loc_predictions
        )
        loc_loss = torch.mean(loc_loss)
        target_loss = torch.sum(is_buggy * -torch.log(target_probs + 1e-9)) / (
            1e-9 + torch.sum(is_buggy)
        )
        return {"loc_loss": loc_loss, "target_loss": target_loss}

    def test_get_acs(
        self,
        is_buggy: torch.tensor,
        loc_predictions: torch.tensor,
        target_probs: torch.tensor,
        error_locations: torch.tensor,
    ) -> dict[str, torch.tensor]:
        rep_accs = (target_probs >= 0.5).type(torch.FloatTensor).to(self._device)
        loc_accs = sparse_categorical_accuracy(error_locations, loc_predictions)
        no_bug_pred_acc = torch.sum((1 - is_buggy) * loc_accs) / (
            1e-9 + torch.sum(1 - is_buggy)
        )
        if torch.sum(1 - is_buggy) == 0:
            no_bug_pred_acc = 1
        bug_loc_acc = torch.sum(is_buggy * loc_accs) / (1e-9 + torch.sum(is_buggy))
        target_loc_acc = torch.sum(is_buggy * rep_accs) / (1e-9 + torch.sum(is_buggy))
        joint_acc = torch.sum(is_buggy * loc_accs * rep_accs) / (
            1e-9 + torch.sum(is_buggy)
        )
        return {
            "no_bug_pred_acc": no_bug_pred_acc,
            "bug_loc_acc": bug_loc_acc,
            "target_loc_acc": target_loc_acc,
            "joint_acc": joint_acc,
        }
