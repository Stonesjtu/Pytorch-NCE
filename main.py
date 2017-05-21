#!/usr/bin/env python
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import data
import model
import nce

def setup_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--prof', action='store_true',
                        help='runs a few iteration with profiling tools')
    parser.add_argument('--nce', action='store_true',
                        help='use NCE as loss function')
    parser.add_argument('--noise_ratio', type=int, default=10,
                        help='set the noise ratio of NCE sampling')
    parser.add_argument('--norm_term', type=int, default=9,
                        help='set the log normalization term of NCE sampling')
    return parser


parser = setup_parser()
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(
    args.data,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=args.cuda,
)

print(corpus.train.dataset.dictionary.idx2word[0])


eval_batch_size = args.batch_size

###############################################################################
# Build the model
###############################################################################

# add the representation for padded index
ntokens = len(corpus.train.dataset.dictionary)
print('Vocabulary size is {}'.format(ntokens))
# add one token for padding
model = model.RNNModel(args.model, ntokens, args.emsize,
                       args.nhid, args.nlayers, args.dropout, args.tied)
print(model)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

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

noise = build_unigram_noise(
    torch.FloatTensor(corpus.train.dataset.dictionary.idx2count)
)

if args.cuda:
    noise = noise.cuda()

nce_criterion = nce.NCELoss(
    ntokens=ntokens,
    emb_size=args.emsize,
    noise=noise,
    noise_ratio=args.noise_ratio,
    norm_term=args.norm_term,
)

if args.cuda:
    nce_criterion.cuda()

###############################################################################
# Training code
###############################################################################


def mask_gen(lengths, cuda=False):
    max_len = lengths[0]
    size = len(lengths)
    mask = torch.ByteTensor(size, max_len).zero_()
    if cuda:
        mask = mask.cuda()
    for i in range(size):
        mask[i][:lengths[i]].fill_(1)
    return mask


def corpus_gen(data_batch, cuda=False):
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
    data = Variable(data)
    target = Variable(target)

    return data, target, length


def eval_cross_entropy(output, target, length):
    mask = Variable(mask_gen(length))
    if args.cuda:
        mask = mask.cuda(async=True)
    output = output.masked_select(
        mask.unsqueeze(dim=2).expand_as(output)
    )
    target = target.masked_select(mask)
    cur_loss = criterion(
        output.view(target.size(0), -1),
        target.contiguous(),
    ).data
    return cur_loss[0] * length.sum()


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    eval_loss = 0
    total_length = 0

    for data_batch in data_source:
        data, target, length = corpus_gen(data_batch)

        if args.cuda:
            data = data.contiguous().cuda(async=True)
            target = target.contiguous().cuda(async=True)

        output = model(data, length).contiguous().view(target.numel(), args.emsize)
        output = output.mm(
            nce_criterion.decoder.weight.t()
        ) + nce_criterion.decoder.bias.repeat(1, output.size(0))
        output = output.view(target.size(0), target.size(1), -1)

        eval_loss += eval_cross_entropy(output, target, length)
        total_length += length.sum()

    return math.exp(eval_loss / total_length)


def train():
    params = [
        {'params': model.parameters()},
        {'params': nce_criterion.parameters()},
    ]
    optimizer = optim.SGD(params=params, lr=lr,
                          momentum=0.9, weight_decay=1e-5)
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    for data_batch in corpus.train:
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to
        # start of the dataset.
        if args.prof and batch == 10:
            break
        optimizer.zero_grad()
        data, target, length = corpus_gen(data_batch, args.cuda)
        mask = Variable(mask_gen(length, args.cuda))

        output = model(data, length)
        output = output.masked_select(
            mask.unsqueeze(dim=2).expand_as(output)
        )


        target = target.masked_select(mask)
        loss = nce_criterion(
            output.view(target.size(0), args.emsize),
            target,
        )
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm(nce_criterion.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches'
                  ' | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(corpus.train), lr,
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            print('-' * 87)
        batch += 1


if __name__ == '__main__':

    # Loop over epochs.
    lr = args.lr
    best_val_ppl = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            val_ppl = evaluate(corpus.valid)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s |'
                  'valid ppl {:8.2f}'.format(epoch,
                                             (time.time() - epoch_start_time),
                                             val_ppl))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_ppl or val_ppl < best_val_ppl:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_ppl = val_ppl
            else:
                # Anneal the learning rate if no improvement has been seen in the
                # validation dataset.
                lr /= 2.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_ppl = evaluate(corpus.test)
    print('=' * 89)
    print('| End of training | test ppl {:8.2f}'.format(test_ppl))
    print('=' * 89)
