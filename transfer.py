import torch
from torch.autograd import Function, Variable

CPU = 'cpu'

def transfer_to(tensor, to_device):
    if to_device == CPU:
        return tensor.cpu()
    else:
        return tensor.cuda(to_device)

class Transfer(Function):
    """Pytorch function for autograd of .cuda and .cpu operation

    An autograd enabled version of builtin .cuda and .cpu

    Input:
        - input: input Variable
        - to_device: target device of data transfer, int for GPU
        `transfer.CPU` for CPU

    Return:
        - output: output Variable

    """

    @staticmethod
    def forward(ctx, input, to_device):
        if input.is_cuda:
            ctx.from_device = input.get_device()
        else:
            ctx.from_device = CPU

        return transfer_to(input, to_device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = transfer_to(grad_output, ctx.from_device)
        return (grad_input, None)

transfer = Transfer.apply
