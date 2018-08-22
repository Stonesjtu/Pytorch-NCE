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
            - input: :math:`(N, E)` where `E = output embedding size`
            - target_batch :math:`(N, E, 1+N_r)`where `N = length,
            E = embedding size, N_r = noise ratio`
        """

        # the size will be used to pack the output of indexlinear
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))  # N,E
        target_idx = target_idx.view(-1).unsqueeze(-1)  # N,1
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))  # N,Nr

        indices = torch.cat([target_idx, noise_idx], dim=-1)

        logits = self._compute_sampled_logit(
            indices, input
        ).view(*original_size, -1)

        target_score, noise_score = logits[:, :, 0], logits[:, :, 1:]
        return target_score, noise_score

    def _compute_sampled_logit(self, indices, input):
        """compute the logits of given indices based on input vector

        Args:
            - indices: (N, M) where `N = length, M = samples per input`
            - input: (N, d) where `d = vector dimension`

        Returns:
            - logits: (N, M) the computed logits
        """

        def select(matrix, idx):
            return matrix.index_select(0, idx.view(-1)).view(*idx.size(), -1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        # [] operator is much slower than index_select in pytorch-0.4.0

        input = input.unsqueeze(1)
        target_batch = select(self.weight, indices).transpose(1,2)
        bias = select(self.bias, indices).squeeze(-1).unsqueeze(1)
        logits = torch.baddbmm(1, bias, 1, input, target_batch)
        return logits

    def ce_loss(self, target_idx, input):
        score = F.linear(input, self.weight, self.bias)  # (N, V)
        loss = self.ce(
            score.view(-1, score.size(-1)),
            target_idx.view(-1)
        ).view_as(target_idx)
        return loss
