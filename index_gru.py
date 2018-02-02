"""An indexed module bundle for generic NCE module"""

import torch
import torch.nn as nn

from nce import NCELoss

class IndexGRU(NCELoss):
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

    def __init__(self, ntoken, ninp, nhid, dropout,
                 *args, **kwargs):
        super(IndexGRU, self).__init__(*args, **kwargs)

        self.ntoken = ntoken
        self.nhid = nhid
        self.ninp = ninp

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

    def get_score(self, target_idx, noise_idx, input):

        if not self.nce:
            #TODO: evaluate Perplexity
            raise(NotImplementedError('CE evaluation mode for GRU is not implemented yet'))

        input_emb = self.encoder(input) # (B, N, E)
        # The noise for <s> (sentence start) is non-sense
        rnn_output, _last_hidden = self.rnn(input_emb) # (B, N, H)

        target_score = self.get_target_score(target_idx, input)

        if noise_idx is None:
            return target_score

        noise_score = self.get_noise_score(noise_idx, rnn_output)

        return target_score, noise_score

    def ce_loss(self, target_idx, input):
        """Compute the CrossEntropyLoss of target index given input

        The loss is an approximation to real CrossEntropyLoss. Due to
        the limitation of generic NCE structure, the score among the whole
        vocabulary is intractable to compute.

        Args:
            - target_idx: the batched target index
            - input: batched input

        Returns:
            - output: the loss for each target_idx
        """
        target_score = self.forward(target_idx, None, input) - self.norm_term
        return target_score


    def get_target_score(self, noise_idx, rnn_output):
        """Get the score of target word given supervised context

        Args:
            - target_idx: (B, N) the target word index
            - rnn_output: output of rnn model

        Return:
            - target_score: (B, N) score for target word index
        """
        # there's a time-step shift in the following code.
        # because noise_output goes through one more RNN cell
        effective_rnn_output = rnn_output[:, 1:]
        return self.scorer(effective_rnn_output).squeeze()


    def get_noise_score(self, noise_idx, rnn_output):
        """Get the score of noise given supervised context

        Args:
            - noise_idx: (B, N, N_r) the noise word index
            - rnn_output: output of rnn model

        Return:
            - noise_score: (B, N, N_r) score for noise word index
        """

        noise_emb = self.encoder(noise_idx.view(-1))
        noise_ratio = noise_idx.size(2)

        # rnn_output of </s> is useless for sentence scoring
        batched_rnn_output = rnn_output[:,:-1].unsqueeze(2).expand(
            -1, -1, noise_ratio, -1
        ).contiguous().view(1, -1, self.nhid)

        noise_output, _last_hidden = self.rnn(
            noise_emb.view(-1, 1, self.nhid),
            batched_rnn_output,
        )

        noise_score = self.scorer(noise_output).view_as(noise_idx)
        return noise_score
