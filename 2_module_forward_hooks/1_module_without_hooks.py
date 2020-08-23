import torch
import torch.nn as nn


class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()

    @staticmethod
    def forward(a, b, c):
        return a + b + c


def main():
    sum_net = SumNet()

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(3.0, requires_grad=True)

    d = sum_net(a, b, c)

    print('d:', d)


if __name__ == '__main__':
    main()
