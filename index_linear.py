"""An index linear class for generic NCE module"""

import torch
import torch.nn as nn

class IndexLinear(nn.Linear):
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
    nce = True

    def __init__(self, input_size, output_size):
        super(IndexLinear, self).__init__(input_size, output_size)
        self.reset_parameters()

    def forward(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_batch :math:`(N, E, 1+N_r)`where `N = length, E = embedding size, N_r = noise ratio`
        """

        # flatten the following matrix
        input = input.view(-1, input.size(-1))
        if not self.nce:
            score = super(IndexLinear, self).forward(input) # (N, V)
            return score

        original_size = target_idx.size() # the size will be used to pack the output of indexlinear
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))

        indices = torch.cat([target_idx.unsqueeze(-1), noise_idx], dim=-1)

        # the pytorch's [] operator BP can't correctly
        input = input.unsqueeze(1)
        target_batch = self.weight.index_select(0, indices.view(-1)).view(indices.size(0), indices.size(1), -1).transpose(1,2)
        bias = self.bias.index_select(0, indices.view(-1)).view(indices.size(0), 1, indices.size(1))
        out = torch.baddbmm(1, bias, 1, input, target_batch).squeeze()
        return out.view(*original_size, -1)

    def reset_parameters(self):
        init_range = 0.1
        self.bias.data.fill_(0)
        self.weight.data.uniform_(-init_range, init_range)
