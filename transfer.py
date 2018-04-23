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
        # print(grad_output.size())
        grad_input = transfer_to(grad_output, ctx.from_device)
        return (grad_input, None)

transfer = Transfer.apply


class TransferEmbedding(Function):
    """An embedding module to workaround slow spcadd of pytorch"""

    @staticmethod
    def forward(ctx, weight, idx, lr=1):
        ctx.idx = idx.view(-1)
        ctx.lr = lr
        ctx.weight = weight
        weight.requires_grad = False
        return weight[ctx.idx].view(*idx.size(), -1)

    @staticmethod
    def backward(ctx, grad_output):
        ctx.weight.index_add_(0, ctx.idx, (-ctx.lr) * grad_output.view(-1, grad_output.size(-1)))
        return (None, None, None)

tranfer_embedding = TransferEmbedding.apply
