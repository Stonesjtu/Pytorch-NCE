"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask
from transfer import transfer

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, sparse=True)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input_emb):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(input_emb)
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        return output


    def forward(self, sentences, length):

        emb_gpu = transfer(self.encoder(sentences), 0)
        mask = get_mask(length.data, max_len=sentences.size(1) - 1)
        input_emb = emb_gpu[:, :-1].contiguous()
        target_emb = emb_gpu[:, 1:].contiguous()
        rnn_output = self._rnn(input_emb)
        loss = self.criterion(sentences[:, 1:].contiguous().cuda(), rnn_output, target_emb)
        loss = torch.masked_select(loss, mask)

        return loss.mean()
