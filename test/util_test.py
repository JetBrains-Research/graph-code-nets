import torch
import tensorflow as tf
from running.util import (
    sparse_categorical_accuracy,
    sparse_softmax_cross_entropy_with_logits,
)


print(
    sparse_categorical_accuracy(
        torch.tensor([2, 1]), torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    )
)

logits = torch.FloatTensor(
    [[2.0, -5.0, 0.5, -0.1], [0.0, 0.0, 1.9, 1.4], [-100.0, 100.0, -100.0, -100.0]]
)
labels = torch.FloatTensor([0, 3, 1])

print(sparse_softmax_cross_entropy_with_logits(labels, logits))
indices = torch.tensor([4, 3, 1, 7])
updates = torch.tensor([9, 10, 11, 12])
shape = torch.tensor([8])
print(torch.zeros(8, dtype=updates.dtype).scatter_(dim=0, index=indices, src=updates))
