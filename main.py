import torch

print(torch.cuda.is_available())

print(torch.tensor([1.0, 2.0]).cuda())
