"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, sparse=True)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        # Usually we use the same # dim in both input and output embedding
        self.proj = nn.Linear(nhid, ninp)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input, hid):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb, hid)
        output = self.proj(output)
        output = self.drop(output)
        return output, unused_hidden

    def forward(self, input, target, length, hid):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output, hid = self._rnn(input, hid)
        loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)

        return loss.mean(), (hid[0].detach(), hid[1].detach())
