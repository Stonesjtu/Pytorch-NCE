"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask
from basis_embedding import BasisEmbedding

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = BasisEmbedding(ntoken, ninp, num_basis=4, num_clusters=8000)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion

    def _rnn(self, input):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        return output


    def forward(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)

        return loss.mean()
