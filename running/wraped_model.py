import numpy as np
import running.util as util
import torch
import torch.nn.functional as F
from models import linear, rnn
import pytorch_lightning as pl
from running.util import (
    sparse_categorical_accuracy,
    sparse_softmax_cross_entropy_with_logits,
)


class VarMisuseLayer(pl.LightningModule):
    def __init__(self, model_config, training_config, vocab_dim):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.vocab_dim = vocab_dim

        self.embedding = torch.normal(
            mean=0,
            std=self.model_config["base"]["hidden_dim"] ** -0.5,
            size=[self.vocab_dim, self.model_config["base"]["hidden_dim"]],
        ).cuda()
        self.pos_enc = util.positional_encoding(
            self.model_config["base"]["hidden_dim"], 5000
        )

        join_dicts = lambda d1, d2: {**d1, **d2}
        base_config = self.model_config["base"]
        inner_model = self.model_config["configuration"]
        self.prediction = linear.Linear(base_config)
        if inner_model == "rnn":
            self.model = rnn.RNN(
                join_dicts(base_config, self.model_config["rnn"]),
                shared_embedding=self.embedding,
            )
        else:
            raise ValueError("Unknown model component provided:", inner_model)

    def forward(self, tokens, token_mask, edges):
        original_shape = list(
            np.append(np.array(tokens.shape), self.model_config["base"]["hidden_dim"])
        )
        flat_tokens = tokens.type(torch.LongTensor).flatten().cuda()
        subtoken_embeddings = torch.index_select(self.embedding, 0, flat_tokens)
        subtoken_embeddings = torch.reshape(subtoken_embeddings, original_shape)
        subtoken_embeddings *= torch.unsqueeze(torch.clamp(tokens, 0, 1), -1).cuda()
        states = torch.mean(subtoken_embeddings, 2)
        # have to understand why the following line is needed
        # states += self.pos_enc[:states.shape[1]]
        predictions = torch.transpose(self.prediction(self.model(states)[0]), 1, 2)
        return predictions

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        tokens, edges, error_loc, repair_targets, repair_candidates = batch
        token_mask = torch.clamp(torch.sum(tokens, -1), 0, 1)
        pointer_preds = self(tokens, token_mask, edges)
        ls, acs = self.get_loss(
            pointer_preds, token_mask, error_loc, repair_targets, repair_candidates
        )
        loss = sum(ls)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, edges, error_loc, repair_targets, repair_candidates = batch
        token_mask = torch.clamp(torch.sum(tokens, -1), 0, 1)
        pointer_preds = self(tokens, token_mask, edges)
        ls, acs = self.get_loss(
            pointer_preds, token_mask, error_loc, repair_targets, repair_candidates
        )
        loss = sum(ls)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.training_config["learning_rate"]
        )

    # probably there are lots of bugs here right now...
    def get_loss(
        self,
        predictions,
        token_mask,
        error_locations,
        repair_targets,
        repair_candidates,
    ):
        seq_mask = token_mask.float()
        predictions += (1.0 - torch.unsqueeze(seq_mask, 1)) * torch.finfo(
            torch.float32
        ).min
        is_buggy = torch.clamp(error_locations, 0, 1).float()
        loc_predictions = predictions[:, 0]
        loc_loss = sparse_softmax_cross_entropy_with_logits(
            error_locations, loc_predictions
        )
        loc_loss = torch.mean(loc_loss)
        loc_accs = sparse_categorical_accuracy(error_locations, loc_predictions)
        no_bug_pred_acc = torch.sum((1 - is_buggy) * loc_accs) / (
            1e-9 + torch.sum(1 - is_buggy)
        )
        bug_loc_acc = torch.sum(is_buggy * loc_accs) / (1e-9 + torch.sum(is_buggy))
        pointer_logits = predictions[:, 1]

        # maybe there is an appropriate function in pytorch for that
        candidate_mask = np.zeros(pointer_logits.size())
        for e in repair_candidates:
            candidate_mask[e[0]][e[1]] = 1
        candidate_mask = torch.tensor(candidate_mask).cuda()

        pointer_logits += (1.0 - candidate_mask) * torch.finfo(torch.float32).min
        pointer_probs = F.softmax(pointer_logits, dim=-1)
        target_mask = np.zeros(pointer_probs.size())
        for e in repair_targets:
            target_mask[e[0]][e[1]] = 1
        target_mask = torch.tensor(target_mask).cuda()

        target_probs = torch.sum(target_mask * pointer_probs, -1)
        target_loss = torch.sum(is_buggy * -torch.log(target_probs + 1e-9)) / (
            1e-9 + torch.sum(is_buggy)
        )
        rep_accs = (target_probs >= 0.5).type(torch.FloatTensor).cuda()
        target_loc_acc = torch.sum(is_buggy * rep_accs) / (1e-9 + torch.sum(is_buggy))

        joint_acc = torch.sum(is_buggy * loc_accs * rep_accs) / (
            1e-9 + torch.sum(is_buggy)
        )
        return (loc_loss, target_loss), (
            no_bug_pred_acc,
            bug_loc_acc,
            target_loc_acc,
            joint_acc,
        )
