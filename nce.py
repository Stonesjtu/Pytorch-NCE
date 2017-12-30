"""A generic NCE wrapper which speedup the training and inferencing"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from alias_multinomial import AliasMethod
from index_gru import IndexGRU


class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are two modes in this NCELoss module:
        - nce: enable the NCE approximtion
        - ce: use the original cross entropy as default loss
    They can be switched by calling function `nce()` or `ce()`
    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        nhidden: hidden size of LSTM(a.k.a the output size)
        ntokens: vocabulary size
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        normed_eval: using normalized probability during evaluation
        index_module: a nn module which takes target and noise idx (maybe
        extra parameters) and outputs the corresponding likelihoods.

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: :math:`(B, N)` if `reduce=True`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduce=False` the scalar NCELoss Variable ready for backward,
        else the loss matrix for every individual targets.

    Shape:
    """
    def __init__(self,
                 ntokens,
                 nhidden,
                 noise,
                 noise_ratio=10,
                 norm_term=9,
                 size_average=True,
                 reduce=False,
                 per_word=True,
                 nce=True,
                 ):
        super(NCELoss, self).__init__()

        self.register_buffer('noise', noise)
        self.alias = AliasMethod(noise)
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.ntokens = ntokens
        self.size_average = size_average
        self.reduce = reduce
        self.per_word = per_word
        self.nce = nce
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        self.index_module = IndexLinear(nhidden, ntokens)
        #self.index_module = IndexGRU(ntokens, nhidden, nhidden, 0.5)


    # set the NCE mode for this module. Similar with module.train()/eval()
    def disable_nce(self):
        self.nce = False
        self.index_module.nce = False

    def enable_nce(self):
        self.nce = True
        self.index_module.nce = True

    def nce_mode(self, status):
        if status:
            self.enable_nce()
        else:
            self.disable_nce()

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.nce:

            noise_samples = self.get_noise(batch, max_len)

            # B,N,Nr
            prob_noise = Variable(
                self.noise[noise_samples.data.view(-1)].view_as(noise_samples)
            )
            prob_target_in_noise = Variable(
                self.noise[target.data.view(-1)].view_as(target)
            )

            # (B,N), (B,N,Nr)
            prob_model, prob_noise_in_model = self._get_prob(target, noise_samples, *args, **kwargs)

            loss = self.nce_loss(
                prob_model, prob_noise_in_model,
                prob_noise, prob_target_in_noise,
            )

        else:
            # Fallback into conventional cross entropy
            out = self.index_module(target, None, *args, **kwargs)
            loss = self.ce_loss(out, target.view(-1)).view(batch, max_len)
        # else:
        #     out = self.index_module(target, None, *args, **kwargs)
        #     nll = out.sub(self.norm_term)
        #     loss = -1 * nll.sum()

        if self.reduce:
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        if self.per_word:
            noise_samples = self.alias.draw(
            batch_size,
            max_len,
            self.noise_ratio,
            ).cuda()
        else:
            noise_samples = self.alias.draw(1, max_len, self.noise_ratio).cuda().expand(batch_size, max_len, self.noise_ratio)

        noise_samples = Variable(noise_samples)
        return noise_samples

    def _get_prob(self, target_idx, noise_idx, *args, **kwargs):
        """Get the NCE estimated probability for target and noise

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        probs = self.index_module(target_idx, noise_idx, *args, **kwargs)

        probs = probs.sub(self.norm_term).exp()
        return probs[:, :, 0], probs[:, :, 1:]

    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - prob_model: probability of target words given by the model (RNN)
            - prob_noise_in_model: probability of noise words given by the model
            - prob_noise: probability of noise words given by the noise distribution
            - prob_target_in_noise: probability of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """
        model_loss = torch.log(prob_model / (
            prob_model + self.noise_ratio * prob_target_in_noise
        ))

        noise_loss = torch.sum(
            torch.log((self.noise_ratio * prob_noise) / (prob_noise_in_model + self.noise_ratio * prob_noise)), -1
        ).squeeze()

        loss = - (model_loss + noise_loss)

        return loss



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
