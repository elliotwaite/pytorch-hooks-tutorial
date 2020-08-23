import torch
import torch.nn as nn


class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()

    @staticmethod
    def forward(a, b, c):
        d = a + b + c
        return d


def forward_pre_hook(module, inputs):
    a, b = inputs
    return a + 10, b


def forward_hook(module, inputs, output):
    return output + 100


def main():
    sum_net = SumNet()

    sum_net.register_forward_pre_hook(forward_pre_hook)
    sum_net.register_forward_hook(forward_hook)

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(3.0, requires_grad=True)

    d = sum_net(a, b, c=c)

    print('d:', d)


if __name__ == '__main__':
    main()
