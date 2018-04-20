"""An index linear class for generic NCE module"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nce import NCELoss

class IndexLinear(NCELoss):
    """A linear layer that only decodes the results of provided indices

    Args:
        target_idx: indices of target words
        noise_idx: indices of noise words
        input: input matrix

    Shape:
        - target_idx :math:`(B, N)` where `max(M) <= N` B is batch size
        - noise_idx :math:`(B, N, N_r)` where `max(M) <= N`
        - Input :math:`(B, N, in\_features)`

    Return:
        - target_score :math:`(N, 1)`
        - noise_score :math:`(N, N_r)` the un-normalized score
    """

    def __init__(self, input_size, output_size, *args, **kwargs):
        super(IndexLinear, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.ce = nn.CrossEntropyLoss(reduce=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # initialize the bias with unigram instead of uniform
            self.bias.data = torch.log(self.noise + 1e-10) + self.norm_term

    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        original_size = target_idx.size()
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx.view(-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        target_batch = self.weight.index_select(0, target_idx)  # N X H
        target_bias = self.bias.index_select(0, target_idx)  # N
        target_score = torch.matmul(target_batch.unsqueeze(1), input.unsqueeze(2)).squeeze() + target_bias  # N X 1 X H * N X H X 1

        noise_batch = self.weight.index_select(0, noise_idx)  # Nr X H
        noise_bias = self.bias.index_select(0, noise_idx)  # Nr
        noise_score = torch.matmul(input, noise_batch.t()) + noise_bias.unsqueeze(0)  # N X Nr
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, input):
        score = F.linear(input, self.weight, self.bias) # (N, V)
        loss = self.ce(score.view(-1, score.size(-1)), target_idx.view(-1)).view_as(target_idx)
        return loss
