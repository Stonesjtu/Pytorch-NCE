import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .nce_loss import BACKOFF_PROB
from .index_linear import IndexLinear


class MultiTarget(IndexLinear):
    """A sampled softmax output for multi-target"""
    def forward(self, target, input):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.

        Args:
            - target: `(B, L)`
            - input: `(B, T, H)`
        """

        input = input.repeat(1, 3, 1)
        batch = input.size(0)
        max_len = input.size(1)
        label_len = target.size(1)

        noise_samples = self.get_noise(batch, max_len)
        # noise_samples = target.view(-1).expand(batch, max_len, -1).contiguous()

        # B,N,Nr
        prob_noise = Variable(
            self.noise[noise_samples.data.view(-1)].view_as(noise_samples)
        )
        prob_target_in_noise = Variable(
            self.noise[target.data.view(-1)].view_as(target).unsqueeze(1).repeat(1, max_len, 1)
        )
        print(prob_target_in_noise.size())

        # (B,N), (B,N,Nr)
        prob_model, prob_noise_in_model = self._get_prob(target, noise_samples, input)
        out = self.sampled_log_softmax(prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise)
        target_out = out[..., :label_len]
        import ipdb
        ipdb.set_trace()
        return target_out

    def sampled_log_softmax(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the sampled softmax given input

        Returns:
            - log_softmax: `(B, T, L+N_r)` approximation of the full log_softmax"""
        logits = torch.cat([prob_model, prob_noise_in_model], dim=2).clamp_min(BACKOFF_PROB).log()
        q_logits = torch.cat([prob_target_in_noise, prob_noise], dim=2).clamp_min(BACKOFF_PROB).log()
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        return F.log_softmax(logits, dim=2)

    def _compute_sampled_logit_batched(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        A batched version, it speeds up computation and puts less burden on
        sampling methods.

        Args:
            - target_idx: :math:`B, L` will be flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, T, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, T, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, T, L)` the computed logits of target_idx
            containing the probs of words in each sentence as target.
            - noise_score: :math:`(B, T, N_r)` the computed logits of noise_idx
        """

        original_size = input.size()[:-1]
        noise_idx = noise_idx[0, 0].view(-1)

        # Use efficient method for single target
        target_batch = self.emb(target_idx)  # B, L, E
        target_bias = self.bias(target_idx)  # B, L
        target_score = torch.matmul(target_batch, input.transpose(1, 2)) + target_bias

        noise_batch = self.emb(noise_idx)  # Nr X H
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)  # Nr
        noise_bias = self.bias(noise_idx)  # 1, Nr
        noise_score = torch.matmul(
            input, noise_batch.t()
        ) + noise_bias.t()  # N X Nr

        target_score = target_score.view(*original_size, -1).squeeze()
        noise_score = noise_score.view(*original_size, -1)
        return target_score, noise_score
