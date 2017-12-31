"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask

class GenModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, criterion, dropout=0.2):
        super(GenModel, self).__init__()
        self.criterion = criterion

    def forward(self, input, target, length):

        mask = get_mask(length, cut_tail=0)

        # <s> is non-sense in this model, thus the loss should be
        # masked manually
        effective_target = target[:, 1:].contiguous()
        loss = self.criterion(effective_target, input)
        loss = torch.masked_select(loss, mask)

        return loss.mean()
