# the NCE module written for pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable


class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    NCE is to eliminate the computational cost of softmax
    normalization.
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
        cuda: indicates the usage of cuda kernel
    """
    def __init__(self, noise, noise_ratio, norm_term, ntokens, size_average=True, cuda=False):
        super(NCELoss, self).__init__()

        self.noise = noise
        self.noise_ratio = noise_ratio
        self.norm_term = norm_term
        self.ntokens = ntokens
        self.size_average = size_average
        self.cuda = cuda

    def forward(self, input, target):
        """compute the loss with output and the desired target
        Parameters:
            input: the output of decoder, before softmax. NXV
            target: the supervised training label. N
        Return:
            the scalar NCELoss Variable ready for backward
        """
        assert input.size(0) == target.size(0)
        data_samples = input.gather(dim=1, index=target.unsqueeze(1))
        data_prob = data_samples.sub(self.norm_term).exp()
        noise_samples = torch.multinomial(
            self.noise.data,
            num_samples=target.size(0) * self.noise_ratio,
            replacement=True,
        ).view(target.size(0), self.noise_ratio)
        if self.cuda:
            noise_samples = noise_samples.cuda()

        noise_probs = self.noise[noise_samples.view(-1)].view_as(noise_samples)
        noise_in_data_probs = input.gather(dim=1, index=Variable(noise_samples)).sub(self.norm_term).exp()

        rnn_loss = torch.log(data_prob / (
            data_prob + self.noise_ratio * self.noise[target.data]
        ))

        noise_loss = torch.sum(
            torch.log((self.noise_ratio * noise_probs) / (noise_in_data_probs + self.noise_ratio * noise_probs)) , 1
        )

        loss = -1 * torch.sum(rnn_loss + noise_loss)
        if self.size_average:
            loss = loss / target.size(0)

        return loss
