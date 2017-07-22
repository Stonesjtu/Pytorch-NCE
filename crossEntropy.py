# the NCE module written for pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class CELoss(nn.Module):
    """Cross Entropy loss with a decoder matrix
    The NCELoss contains such a decoder matrix, which makes it
    inconsistent with conventional word level language model. So
    I bundle together CrossEntropyLoss function and decoder matrix.

    Attributes:
        nhidden: hidden size of LSTM(a.k.a the output size)
        ntokens: vocabulary size
        size_average: average the loss by batch size
        decoder: the decoder matrix

    Shape:
        - decoder: :math:`(E, V)` where `E = embedding size`
    """

    def __init__(self,
                 ntokens,
                 nhidden,
                 size_average=True,
                 decoder_weight=None,
                 ):
        super(CELoss, self).__init__()

        self.ntokens = ntokens
        self.size_average = size_average
        self.decoder = nn.Linear(nhidden, ntokens)
        self.criterion = nn.CrossEntropyLoss()
        # Weight tying
        if decoder_weight:
            self.decoder.weight, self.decoder.bias = decoder_weight

    def forward(self, input, target):
        """compute the loss with output and the desired target

        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.

        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`

        Return:
            the scalar NCELoss Variable ready for backward
        """

        output = self.decoder(input)
        loss = self.criterion(output, target)
        return loss

