# Collections of some helper functions
import logging
import argparse

import torch


def setup_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch PennTreeBank NCE Language Model')
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default=None,
                        help='location of the vocabulary file, without which will use vocab of training corpus')
    parser.add_argument('--min-freq', type=int, default=1,
                        help='minimal frequency for word to build vocabulary')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--bptt', type=int, default=35,
                        help='truncated bptt length')
    parser.add_argument('--concat', action='store_true',
                        help='Use concatenated sentences chunked into length of bptt')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='initial weight decay')
    parser.add_argument('--lr-decay', type=float, default=2,
                        help='learning rate decay when no progress is observed on validation set')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--loss', type=str, default='full',
                        help='use one of [nce, full, sampled, mix] as loss '
                        'function')
    parser.add_argument('--index-module', type=str, default='linear',
                        help='index module to use in NCELoss wrapper')
    parser.add_argument('--noise-ratio', type=int, default=50,
                        help='set the noise ratio of NCE sampling, the noise'
                        ' is shared among batch')
    parser.add_argument('--norm-term', type=int, default=-1,
                        help='set the log normalization term of NCE sampling')
    parser.add_argument('--train', action='store_true',
                        help='set train mode, otherwise only evaluation is'
                        ' performed')
    parser.add_argument('--tb-name', type=str, default=None,
                        help='the name which would be used in tensorboard '
                        'record')
    parser.add_argument('--prof', action='store_true',
                        help='Enable profiling mode, will execute only one '
                        'batch data')
    return parser


def setup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('log/%s.log' % logger_name)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# Get the mask matrix of a batched input
def get_mask(lengths, cut_tail=0, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    assert lengths.min() >= cut_tail
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask


def process_data(data_batch, cuda=False, sep_target=True):
    """A data pre-processing util which construct the input `Tensor` for model

    Args:
        - data_batch: a batched data from `PaddedDataset`
        - cuda: indicates whether to put data into GPU
        - sep_target: return separated input and target if turned on

    Returns:
        - input: the input data batch
        - target: target data if `sep_target` is True, else a duplicated input
        - effective_length: the useful sentence length for loss computation <s> is ignored
        """

    batch_sentence, length = data_batch
    if cuda:
        batch_sentence = batch_sentence.cuda()
        length = length.cuda()

    # cut the padded sentence to max sentence length in this batch
    max_len = length.max()
    batch_sentence = batch_sentence[:, :max_len]

    # the useful sentence length for loss computation <s> is ignored
    effective_length = length - 1

    if sep_target:
        data = batch_sentence[:, :-1]
        target = batch_sentence[:, 1:]
    else:
        data = batch_sentence
        target = batch_sentence

    data = data.contiguous()
    target = target.contiguous()
    effective_length = effective_length

    return data, target, effective_length


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
