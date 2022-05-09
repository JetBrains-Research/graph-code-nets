import numpy as np
import models.util as util
import torch
import torch.nn.functional as F
from models import two_pointer_fcn, encoder_gru, encoder_ggnn
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_geometric.data import Data
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
        elif inner_model == "ggsnn":
            pass
        else:
            raise ValueError("Unknown model component provided:", inner_model)

    def forward(  # type: ignore[override]
        self, tokens: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        subtoken_embeddings = self._embedding(tokens) * torch.unsqueeze(
            torch.clamp(tokens, 0, 1), -1
        )
        states = torch.mean(subtoken_embeddings, 2)
        predictions = self._model(states, edges)
        return self._prediction(predictions)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._shared_eval_step(batch, batch_idx, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._shared_eval_step(batch, batch_idx, "test")

    def _shared_eval_step(self, batch: Data, batch_idx: int, step: str) -> torch.Tensor:
        pointer_preds = self(batch.x, batch.edge_index)
        print("pointer_preds", pointer_preds, pointer_preds.size())
        return torch.sum(pointer_preds, (0, 1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self._training_config["learning_rate"]
        )

    def _shared_loss_acs_calc(
        self,
        predictions: torch.Tensor,
        token_mask: torch.Tensor,
        error_locations: torch.Tensor,
        repair_targets: torch.Tensor,
        repair_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_mask = token_mask.float()
        predictions += (1.0 - torch.unsqueeze(seq_mask, 1)) * torch.finfo(
            torch.float32
        ).min
        is_buggy = torch.clamp(error_locations, 0, 1).float()
        loc_predictions = predictions[:, 0]
        pointer_logits = predictions[:, 1]

        candidate_mask = torch.zeros(pointer_logits.size())
        for e in repair_candidates:
            candidate_mask[e[0]][e[1]] = 1
        candidate_mask = candidate_mask.to(self._device)

        pointer_logits += (1.0 - candidate_mask) * torch.finfo(torch.float32).min
        pointer_probs = F.softmax(pointer_logits, dim=-1)

        target_mask = torch.zeros(pointer_probs.size())
        for e in repair_targets:
            target_mask[e[0]][e[1]] = 1
        target_mask = target_mask.to(self._device)
        target_probs = torch.sum(target_mask * pointer_probs, -1)

        return is_buggy, loc_predictions, target_probs

    def test_get_losses(
        self,
        is_buggy: torch.Tensor,
        loc_predictions: torch.Tensor,
        target_probs: torch.Tensor,
        error_locations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loc_loss = sparse_softmax_cross_entropy_with_logits(
            error_locations, loc_predictions
        )
        loc_loss = torch.mean(loc_loss)
        target_loss = torch.sum(is_buggy * -torch.log(target_probs + 1e-9)) / (
            1e-9 + torch.sum(is_buggy)
        )
        return {"loc_loss": loc_loss, "target_loss": target_loss}

    def test_get_accuracies(
        self,
        is_buggy: torch.Tensor,
        loc_predictions: torch.Tensor,
        target_probs: torch.Tensor,
        error_locations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rep_accuracies = (target_probs >= 0.5).type(torch.float32).to(self._device)
        loc_accuracies = sparse_categorical_accuracy(error_locations, loc_predictions)
        no_bug_pred_acc = torch.sum((1 - is_buggy) * loc_accuracies) / (
            1e-9 + torch.sum(1 - is_buggy)
        )
        if torch.sum(1 - is_buggy) == 0:
            no_bug_pred_acc = torch.tensor(1)
        bug_loc_acc = torch.sum(is_buggy * loc_accuracies) / (
            1e-9 + torch.sum(is_buggy)
        )
        target_loc_acc = torch.sum(is_buggy * rep_accuracies) / (
            1e-9 + torch.sum(is_buggy)
        )
        joint_acc = torch.sum(is_buggy * loc_accuracies * rep_accuracies) / (
            1e-9 + torch.sum(is_buggy)
        )
        return {
            "no_bug_pred_acc": no_bug_pred_acc,
            "bug_loc_acc": bug_loc_acc,
            "target_loc_acc": target_loc_acc,
            "joint_acc": joint_acc,
        }
