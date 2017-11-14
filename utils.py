# Collections of some helper functions
import torch
from torch.autograd import Variable


# Get the mask matrix of a batched input
def get_mask(lengths):
    max_len = lengths[0]
    size = len(lengths)
    mask = lengths.new().byte().resize_(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return Variable(mask)


def process_data(data_batch, cuda=False, eval=False):
    data, target, length = data_batch

    if cuda:
        data = data.cuda()
        target = target.cuda()
        length = length.cuda()

    length, idx = torch.sort(length, dim=0, descending=True)
    max_len = length[0]
    data = data.index_select(0, idx)
    data = data[:, :max_len]
    target = target.index_select(0, idx)
    target = target[:, :max_len]
    data = Variable(data, volatile=eval)
    target = Variable(target)
    return data, target, length


def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001
    return noise
