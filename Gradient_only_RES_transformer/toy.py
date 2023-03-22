# import torch
# import torch.nn as nn

# a = torch.tensor([[1, 2, 1], [1, 1, 1]])
# print(a.shape)
# b = a.transpose(0, 1)
# print(a)
# print(b)
# c = a @ b
# print(c)

# a = nn.Linear(120, 100)
# model = a
# # count the number of parameters in the model
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# params = [p for p in model.parameters()]
# print(f"Number of parameters: {num_params}")
# print(len(params))
# print(len(params[0]))
# print(len(params[1]))


# import torch
# import torch.nn as nn


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.linear1 = nn.Linear(10, 5)
#         # self.linear2 = nn.Linear(5, 1)

#     def forward(self, x):
#         x = torch.relu(self.linear1(x))
#         # x = self.linear2(x)
#         return x


# model = MyModel()

# # count the number of parameters in the model
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {num_params}")


####################################
import torch

a = torch.tensor([1, 2, 3, 4, 5])
print(a)

b = torch.rand(3, 5)
print(b.shape)
print(b)

c = torch.zeros(1)
print(c * b)
print()


# import torch

# # Create a tensor
# # x = torch.tensor([2.0, 3.0], requires_grad=True)
# x = torch.tensor([3.0], requires_grad=True)

# # Perform some computations using the tensor
# # y = x.sum()
# # z = y * 2
# y = x * 0


# # Compute the gradient of z with respect to x
# y.backward()

# # Print the gradient of x
# print(x.grad)
