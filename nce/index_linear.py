"""An index linear class for generic NCE module"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nce_loss import NCELoss


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

    def __init__(self, embedding_dim, num_classes, *args, **kwargs):
        super(IndexLinear, self).__init__(*args, **kwargs)
        # use Embedding to store the output embedding
        # it's efficient when it comes sparse update of gradients
        self.emb = nn.Embedding(num_classes, embedding_dim)
        # self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.bias = nn.Embedding(num_classes, 1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.emb.embedding_dim)
        self.emb.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # initialize the bias with unigram instead of uniform
            self.bias.weight.data = torch.unsqueeze(
                torch.log(self.noise + 1e-10) + self.norm_term, 1
            )

    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_idx: :math:`(B, L)` where `B` is batch size
            `L` is sequence length
            - noise_idx: :math:`(B, L, N_r)` where `N_r is noise ratio`
            - input: :math:`(B, L, E)` where `E = output embedding size`
        """

        if self.per_word:
            return self._compute_sampled_logit(
                target_idx, noise_idx, input
            )
        else:
            return self._compute_sampled_logit_batched(
                target_idx, noise_idx, input
            )

    def _compute_sampled_logit(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        Args:
            - target_idx: :math:`B, L, 1`
            - noise_idx: :math:`B, L, N_r` target_idx and noise_idx are
            concatenated into one single index matrix for performance
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        # the size will be used to pack the output of indexlinear
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, 1, input.size(-1))  # N,1,E
        target_idx = target_idx.view(-1).unsqueeze(-1)  # N,1
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))  # N,Nr
        indices = torch.cat([target_idx, noise_idx], dim=-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        # [] operator is much slower than index_select in pytorch-0.4.0

        # index_select is faster than pure embedding look-up which is weird
        # 20it/s vs. 14 it/s
        # target_batch = self.emb(indices)
        # bias = self.bias(indices).squeeze(2)
        target_batch = self.emb.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1)
        bias = self.bias.weight.index_select(0, indices.view(-1)).view_as(indices)
        # the element-wise multiplication is automatically broadcasted
        logits = torch.sum(input * target_batch, dim=2) + bias
        logits = logits.view(*original_size, -1)

        target_score, noise_score = logits[:, :, 0], logits[:, :, 1:]
        return target_score, noise_score

    def _compute_sampled_logit_batched(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        A batched version, it speeds up computation and puts less burden on
        sampling methods.

        Args:
            - target_idx: :math:`B, L, 1` flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, L, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)

        target_batch = self.emb(target_idx)
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_bias = self.bias(target_idx).squeeze(1)  # N
        target_score = torch.sum(input * target_batch, dim=1) + target_bias  # N X E * N X E

        noise_batch = self.emb(noise_idx)  # Nr X H
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)  # Nr
        noise_bias = self.bias(noise_idx)  # 1, Nr
        noise_score = torch.matmul(
            input, noise_batch.t()
        ) + noise_bias.t()  # N X Nr
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, input):
        score = F.linear(input, self.emb.weight, self.bias.weight.squeeze(1))  # (N, V)
        loss = self.ce(
            score.view(-1, score.size(-1)),
            target_idx.view(-1)
        ).view_as(target_idx)
        return loss
