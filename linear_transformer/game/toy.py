import torch

a = torch.zeros(1, 5)

print(a)
a[:, :3] = 1

print(a)
