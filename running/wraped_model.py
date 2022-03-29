import util
import torch
import torch.nn.functional as F
from models import rnn
import pytorch_lightning as pl
from util import sparse_categorical_accuracy, sparse_softmax_cross_entropy_with_logits


class VarMisuseLayer(pl.LightningModule):

    def __init__(self, model_config, training_config, vocab_dim):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.vocab_dim = vocab_dim

        self.embedding = torch.normal(mean=0, std=self.model_config['base']['hidden_dim'] ** -0.5,
                                      size=[self.vocab_dim, self.model_config['base']['hidden_dim']])
        self.prediction = torch.nn.Linear(self.model_config['base']['hidden_dim'], 2)
        self.pos_enc = util.positional_encoding(self.model_config['base']['hidden_dim'], 5000)

        join_dicts = lambda d1, d2: {**d1, **d2}
        base_config = self.model_config['base']
        inner_model = self.model_config['configuration']
        if inner_model == 'rnn':
            self.model = rnn.RNN(join_dicts(base_config, self.model_config['rnn']), shared_embedding=self.embedding)
        else:
            raise ValueError('Unknown model component provided:', inner_model)

    def forward(self, tokens, token_mask, edges, training):
        subtoken_embeddings = torch.index_select(self.embedding, 0, tokens)
        subtoken_embeddings *= torch.unsqueeze(torch.clamp(tokens, 0, 1), -1)
        states = torch.mean(subtoken_embeddings, 2)
        states = self.model(states, training=training)
        predictions = torch.transpose(self.prediction(states), 1, 2)
        return predictions

    def training_step(self, batch, batch_idx):
        tokens, edges, error_loc, repair_targets, repair_candidates = batch
        token_mask = torch.clamp(torch.sum(tokens, -1), 0, 1)

        pointer_preds = self(tokens, token_mask, edges, training=True)
        ls, acs = self.get_loss(pointer_preds, token_mask, error_loc, repair_targets, repair_candidates)
        loss = sum(ls)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.training_config['learning_rate'])

    def get_loss(self, predictions, token_mask, error_locations, repair_targets, repair_candidates):
        seq_mask = token_mask.float()
        predictions += (1.0 - torch.unsqueeze(seq_mask, 1)) * torch.finfo(torch.float32).min
        is_buggy = torch.clamp(error_locations, 0, 1).float()

        loc_predictions = predictions[:, 0]
        # double check
        loc_loss = sparse_categorical_accuracy(error_locations, loc_predictions)
        loc_loss = torch.mean(loc_loss)
        loc_accs = sparse_categorical_accuracy(error_locations, loc_predictions)

        # Store two metrics: the accuracy at predicting specifically the non-buggy samples correctly (to measure false alarm rate), and the accuracy at detecting the real bugs.
        no_bug_pred_acc = torch.sum((1 - is_buggy) * loc_accs) / (
                1e-9 + torch.sum(1 - is_buggy))  # Take mean only on sequences without errors
        bug_loc_acc = torch.sum(is_buggy * loc_accs) / (1e-9 + torch.sum(is_buggy))  # Only on errors

        # For repair accuracy, first convert to probabilities, masking out any non-candidate tokens
        pointer_logits = predictions[:, 1]
        # double check
        candidate_mask = torch.zeros(pointer_logits.size(), dtype=repair_candidates.dtype). \
            scatter_(dim=0, index=repair_candidates, src=torch.ones(repair_candidates.size()[0]))
        pointer_logits += (1.0 - candidate_mask) * torch.finfo(torch.float32).min
        pointer_probs = F.softmax(pointer_logits)

        # Aggregate probabilities at repair targets to get the sum total probability assigned to the correct variable name
        target_mask = torch.zeros(pointer_probs.size(), dtype=repair_targets.dtype). \
            scatter_(dim=0, index=repair_targets, src=torch.ones(repair_targets.size()[0]))
        target_probs = torch.sum(target_mask * pointer_probs, -1)

        # The loss is only computed at buggy samples, using (negative) cross-entropy
        target_loss = torch.sum(is_buggy * -torch.log(target_probs + 1e-9)) / (
                1e-9 + torch.sum(is_buggy))  # Only on errors

        # To simplify the comparison, accuracy is computed as achieving >= 50% probability for the top guess
        # (as opposed to the slightly more accurate, but hard to compute quickly, greatest probability among distinct variable names).
        rep_accs = (target_probs >= 0.5).type(torch.FloatTensor)
        target_loc_acc = torch.sum(is_buggy * rep_accs) / (1e-9 + torch.sum(is_buggy))  # Only on errors

        # Also store the joint localization and repair accuracy -- arguably the most important metric.
        joint_acc = torch.sum(is_buggy * loc_accs * rep_accs) / (1e-9 + torch.sum(is_buggy))  # Only on errors
        return (loc_loss, target_loss), (no_bug_pred_acc, bug_loc_acc, target_loc_acc, joint_acc)
