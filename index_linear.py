"""An index linear class for generic NCE module"""

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

    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        original_size = target_idx.size() # the size will be used to pack the output of indexlinear
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))

        indices = torch.cat([target_idx.unsqueeze(-1), noise_idx], dim=-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        input = input.unsqueeze(1)
        target_batch = self.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1).transpose(1,2)
        bias = self.bias.index_select(0, indices.view(-1)).view_as(indices).unsqueeze(1)
        out = torch.baddbmm(1, bias, 1, input, target_batch).view(*original_size, -1)
        target_score, noise_score = out[:, :, 0], out[:, :, 1:]
        return target_score, noise_score

    def ce_loss(self, target_idx, input):
        score = F.linear(input, self.weight, self.bias) # (N, V)
        loss = self.ce(score.view(-1, score.size(-1)), target_idx.view(-1)).view_as(target_idx)
        return loss
