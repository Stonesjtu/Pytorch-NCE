import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from utils import get_mask

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input, lengths):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(self.encoder(input))
        if lengths is not None:
            emb = pack_padded_sequence(emb, list(lengths), batch_first=True)
        output, unused_hidden = self.rnn(emb)
        if isinstance(output, PackedSequence):
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.drop(output)
        return output

    def _mask(self, padded_output, padded_target, lengths):
        '''mask the padded part introduced by batching sentences
        of variable lengths'''
        mask = get_mask(lengths)
        target = torch.masked_select(padded_target, mask)
        output = torch.masked_select(
            padded_output,
            mask.unsqueeze(2).expand_as(padded_output),
        )
        return output, target

    def _loss(self, flatten_output, target):
        '''Serves as the decoder and the loss framework

        They are integrated into one single criterion module, which
        simplifies the API of NCE loss and normal CE loss
        '''
        loss = self.criterion(flatten_output, target)
        return loss

    def forward(self, input, target, lengths=None):

        rnn_output = self._rnn(input, lengths)
        rnn_output, target = self._mask(rnn_output, target, lengths)
        flatten_output = rnn_output.view(target.size(0), self.nhid)
        loss = self._loss(flatten_output, target)
        return loss
