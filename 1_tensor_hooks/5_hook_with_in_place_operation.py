import torch


a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b
c.retain_grad()
d = torch.tensor(4.0, requires_grad=True)


def d_hook(grad):
    grad *= 100


d.register_hook(d_hook)

e = c + d
e.retain_grad()
e.backward()

print(f'a.grad {a.grad}')
print(f'b.grad {b.grad}')
print(f'c.grad {c.grad}')
print(f'd.grad {d.grad}')
print(f'e.grad {e.grad}')