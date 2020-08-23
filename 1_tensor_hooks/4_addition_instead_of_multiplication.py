import torch


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b

d = torch.tensor(4.0, requires_grad=True)

e = c + d

e.backward()
