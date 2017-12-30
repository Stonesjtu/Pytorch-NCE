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
                 index_module,
                 noise,
                 noise_ratio=10,
                 norm_term=9,
                 size_average=True,
                 reduce=False,
                 per_word=True,
                 nce=True,
                 ):
        super(NCELoss, self).__init__()

        self.index_module = index_module
        self.register_buffer('noise', noise)
        self.alias = AliasMethod(noise)
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.size_average = size_average
        self.reduce = reduce
        self.per_word = per_word
        self.nce = nce
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)


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

