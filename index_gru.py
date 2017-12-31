"""An indexed module bundle for generic NCE module"""

import torch
import torch.nn as nn

class IndexGRU(nn.Module):
    """An indexed module for generic NCE

    This module is a container of nn.Embedding,
    one layer GRU and a linear regressor.

    Attributes:
        - ntoken: size of vocabulary
        - ninp: embedding dimension, also the input dimension of GRU
        - nhid: GRU size
        - dropout: dropout rate for Embedding, GRU and Linear module
        the GRU is not dropout due to only one layer.

    Parameters:
        - target_idx:(B, N) padded target index
        - noise_idx:(B, N, Nr) padded noise index
    """
    nce = True

    def __init__(self, ntoken, ninp, nhid, dropout=0.2):
        super(IndexGRU, self).__init__()

        self.nhid = nhid
        self.ninp = ninp

        dropout = 0.2
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            nn.Embedding(ntoken, ninp),
            self.drop,
        )
        # this GRU only outputs the hidden for last layer, so only 1 layer is supported
        self.rnn = nn.GRU(ninp, nhid, num_layers=1, dropout=dropout, batch_first=True)
        self.scorer = nn.Sequential(
            self.drop,
            nn.Linear(nhid, 1),
        )

    def forward(self, target_idx, noise_idx, input):
        input_emb = self.encoder(input) # (B, N, E)
        # The noise for <s> (sentence start) is non-sense
        noise_emb = self.encoder(noise_idx.view(-1))

        rnn_output, _last_hidden = self.rnn(input_emb) # (B, N, H)
        # there's a time-step shift in the following code.
        # because neg_output goes through one more RNN cell
        effective_rnn_output = rnn_output[:, 1:]
        batched_rnn_output = effective_rnn_output.unsqueeze(2).expand(
            -1, -1, noise_idx.size(2), -1
        ).contiguous().view(1, -1, self.nhid)

        neg_output, _last_hidden = self.rnn(
            noise_emb.view(-1, 1, self.nhid),
            batched_rnn_output,
        )

        target_score = self.scorer(effective_rnn_output).squeeze()
        noise_score = self.scorer(neg_output).view_as(noise_idx)

        return target_score, noise_score
