"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        return output


    def forward(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        likelihood = self.decoder(rnn_output.contiguous().view(-1, rnn_output.size(-1)))
        loss = self.criterion(likelihood, target.view(-1)).view_as(target)
        loss = torch.masked_select(loss, mask)

        return loss.mean()
