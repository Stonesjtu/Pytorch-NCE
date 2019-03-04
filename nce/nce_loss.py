"""A generic NCE wrapper which speedup the training and inferencing"""

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from .alias_multinomial import AliasMultinomial


# A backoff probability to stabilize log operation
BACKOFF_PROB = 1e-10

class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are two modes in this NCELoss module:
        - nce: enable the NCE approximtion
        - ce: use the original cross entropy as default loss
    They can be switched by calling function `enable_nce()` or
    `disable_nce()`, you can also switch on/off via `nce_mode(True/False)`

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: :math:`(B, N)` if `reduce=True`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss Variable ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 ):
        super(NCELoss, self).__init__()

        self.register_buffer('noise', noise)
        self.noise.clamp_min_(BACKOFF_PROB)
        self.alias = AliasMultinomial(noise)
        self.noise_ratio = noise_ratio
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce = nn.BCELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

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

            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        prob_model, prob_noise_in_model,
                        prob_noise, prob_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - prob_model.log()
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    prob_model, prob_noise_in_model,
                    prob_noise, prob_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError('loss type {} not implemented at {}'.format(self.loss_type, current_stage))

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
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
            )
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(batch_size, max_len, self.noise_ratio)

        noise_samples = Variable(noise_samples).contiguous()
        return noise_samples

    def _get_prob(self, target_idx, noise_idx, *args, **kwargs):
        """Get the NCE estimated probability for target and noise

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_score, noise_score = self.get_score(target_idx, noise_idx, *args, **kwargs)

        target_prob = target_score.sub(self.norm_term).exp()
        noise_prob = noise_score.sub(self.norm_term).exp()
        return target_prob, noise_prob

    def get_score(self, target_idx, noise_idx, *args, **kwargs):
        """Get the target and noise scores given input

        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        raise NotImplementedError()

    def ce_loss(self, target_idx, *args, **kwargs):
        """Get the conventional CrossEntropyLoss

        The returned loss should be of the same size of `target`

        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class

        Returns:
            - loss: the estimated loss for each target
        """
        raise NotImplementedError()

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

        p_model = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2).clamp(BACKOFF_PROB, 1)
        p_noise = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2).clamp(BACKOFF_PROB, 1)

        # predicted probability of the word comes from true data distribution
        p_true = p_model / (p_model + self.noise_ratio * p_noise)
        label = torch.cat(
            [torch.ones_like(prob_model).unsqueeze(2),
             torch.zeros_like(prob_noise)], dim=2
        )

        loss = self.bce(p_true, label).sum(dim=2)

        return loss

    def sampled_softmax_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        logits = torch.cat([prob_model.unsqueeze(2), prob_noise_in_model], dim=2).clamp_min(BACKOFF_PROB).log()
        q_logits = torch.cat([prob_target_in_noise.unsqueeze(2), prob_noise], dim=2).clamp_min(BACKOFF_PROB).log()
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        loss = self.ce(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)

        return loss
