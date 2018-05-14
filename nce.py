"""A generic NCE wrapper which speedup the training and inferencing"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from alias_multinomial import AliasMethod


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
        norm_term: the normalization term (lnZ in paper)
        size_average: average the loss by batch size
        reduce: returned the loss for each target_idx if True,
        this will ignore the value of `size_average`

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

    nce = True

    def __init__(self,
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
        self.size_average = size_average
        self.reduce = reduce
        self.per_word = per_word
        self.nce = nce

    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
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
            loss = self.ce_loss(target, *args, **kwargs)

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
            )
        else:
            noise_samples = self.alias.draw(1, max_len, self.noise_ratio).expand(batch_size, max_len, self.noise_ratio)

        noise_samples = Variable(noise_samples)
        return noise_samples

    def _get_prob(self, target_idx, noise_idx, *args, **kwargs):
        """Get the NCE estimated probability for target and noise

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_prob, noise_prob = self.get_score(target_idx, noise_idx, *args, **kwargs)

        target_prob = target_prob.sub(self.norm_term).exp()
        noise_prob = noise_prob.sub(self.norm_term).exp()
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
        def safe_log(tensor):
            """A wrapper to compute logarithm

            An epsilon is pre added for the sake of numeric stability

            Args:
                - tensor: a pytorch Tensor or Variable
            """
            EPSILON = 1e-10
            return torch.log(EPSILON + tensor)

        model_loss = safe_log(prob_model / (
            prob_model + self.noise_ratio * prob_target_in_noise
        ))

        noise_loss = torch.sum(
            safe_log((self.noise_ratio * prob_noise) / (prob_noise_in_model + self.noise_ratio * prob_noise)), -1
        ).squeeze()

        loss = - (model_loss + noise_loss)

        return loss
