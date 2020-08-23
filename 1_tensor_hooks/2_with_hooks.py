import torch


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b


def c_hook(grad):
    print(grad)
    return grad + 2


c.register_hook(c_hook)
c.register_hook(lambda grad: print(grad))
c.retain_grad()

d = torch.tensor(4.0, requires_grad=True)
d.register_hook(lambda grad: grad + 100)

e = c * d

e.retain_grad()
e.register_hook(lambda grad: grad * 2)
e.retain_grad()

e.backward()
