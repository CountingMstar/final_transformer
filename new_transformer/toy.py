import torch

a = torch.tensor([[1, 2, 1], [1, 1, 1]])
print(a.shape)
b = a.transpose(0, 1)
print(a)
print(b)
c = a @ b
print(c)
