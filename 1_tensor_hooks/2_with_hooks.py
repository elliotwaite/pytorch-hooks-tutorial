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
e.retain_grad() # second-time retain_grad() is NOP

e.backward()

print(f'a.grad {a.grad}')
print(f'b.grad {b.grad}')
print(f'c.grad {c.grad}')
print(f'd.grad {d.grad}')
print(f'e.grad {e.grad}')