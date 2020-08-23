import torch
import torch.nn as nn


class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()

    @staticmethod
    def forward(a, b, c):
        d = a + b + c

        print('forward():')
        print('    a:', a)
        print('    b:', b)
        print('    c:', c)
        print()
        print('    d:', d)
        print()

        return d


def forward_pre_hook(module, input_positional_args):
    a, b = input_positional_args
    new_input_positional_args = a + 10, b

    print('forward_pre_hook():')
    print('    module:', module)
    print('    input_positional_args:', input_positional_args)
    print()
    print('    new_input_positional_args:', new_input_positional_args)
    print()

    return new_input_positional_args


def forward_hook(module, input_positional_args, output):
    new_output = output + 100

    print('forward_hook():')
    print('    module:', module)
    print('    input_positional_args:', input_positional_args)
    print('    output:', output)
    print()
    print('    new_output:', new_output)
    print()

    return new_output


def main():
    sum_net = SumNet()
    sum_net.register_forward_pre_hook(forward_pre_hook)
    sum_net.register_forward_hook(forward_hook)

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(3.0, requires_grad=True)

    print('start')
    print()
    print('a:', a)
    print('b:', b)
    print('c:', c)
    print()
    print('before model')
    print()

    d = sum_net(a, b, c=c)

    print('after model')
    print()
    print('d:', d)


if __name__ == '__main__':
    main()
