import torch


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b

d = torch.tensor(4.0, requires_grad=True)


def d_hook(grad):
    grad *= 100


d.register_hook(d_hook)

e = c + d

e.backward()
