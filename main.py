#!/usr/bin/env python

import sys
import time
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import data
from model import RNNModel
from nce import NCELoss
from utils import process_data, build_unigram_noise, setup_parser, setup_logger
from generic_model import GenModel
from index_gru import IndexGRU
from index_linear import IndexLinear
from basis_embedding import BasisEmbedding


parser = setup_parser()
args = parser.parse_args()
logger = setup_logger('pt-nce-%s' % args.save)
logger.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.warning('You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args.seed)

#################################################################
# Load data
#################################################################
corpus = data.Corpus(
    path=args.data,
    vocab_path=args.vocab,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=args.cuda,
    min_freq=args.min_freq,
)

################################################################## Build the criterion and model, setup the NCE module
#################################################################

ntoken = len(corpus.train.dataset.vocab)
logger.info('Vocabulary size is {}'.format(ntoken))

def setup_model(ntoken, args):
    """Setup the model by the arguments passed to main"""
    # noise for soise sampling in NCE
    noise = build_unigram_noise(
        torch.FloatTensor(corpus.train.dataset.vocab.idx2count)
    )
    if args.index_module == 'linear':
        criterion = IndexLinear(
            args.nhid,
            ntoken,
            noise=noise,
            noise_ratio=args.noise_ratio,
            norm_term=args.norm_term,
        )
        criterion.nce = args.nce
        model = RNNModel(
            ntoken, args.emsize, args.nhid, args.nlayers,
            criterion=criterion, dropout=args.dropout,
        )
        sep_target = True

    elif args.index_module == 'gru':
        logger.warning('Falling into one layer GRU due to indx_GRU supporting')
        nce_criterion = IndexGRU(
            ntoken, args.nhid, args.nhid,
            args.dropout,
            noise=noise,
            noise_ratio=args.noise_ratio,
            norm_term=args.norm_term,
        )
        model = GenModel(
            criterion=nce_criterion,
            dropout=args.dropout,
        )

    else:
        logger.error('The index module [%s] is not supported yet' % args.index_module)
        raise NotImplementedError('index module not supported')

    # build the encoder for BasisEmbedding (probably)
    bs_embedding = BasisEmbedding(ntoken, args.emsize, args.num_basis, args.num_cluster)
    if args.index_module == 'linear':
        model.encoder = bs_embedding
    else:
        model.criterion.index_module.encoder[0] = bs_embedding

    if args.cuda:
        model.cuda()

    return model, bs_embedding, sep_target

model, encoder, sep_target = setup_model(ntoken, args)

logger.info('model definition:\n %s', model)
#################################################################
# Training code
#################################################################


def train(model, data_source, epoch, lr=1.0, weight_decay=1e-5, momentum=0.9):
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    # Turn on training mode which enables dropout.
    model.train()
    model.criterion.nce = args.nce
    p_model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    total_loss = 0
    pbar = tqdm(data_source, desc='Training PPL: ....')
    for num_batch, data_batch in enumerate(pbar):
        optimizer.zero_grad()
        data, target, length = process_data(data_batch, cuda=args.cuda, sep_target=sep_target)
        loss = p_model(data, target, length).mean()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data[0]

        if num_batch % args.log_interval == 0 and num_batch > 0:
            if args.prof:
                break
            cur_loss = total_loss / args.log_interval
            ppl = math.exp(cur_loss)
            logger.debug(
                '| epoch {:3d} | {:5d}/{:5d} batches '
                '| lr {:02.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, num_batch, len(corpus.train),
                    lr, cur_loss, ppl
                  )
            )
            pbar.set_description('Training PPL %.1f' % ppl)
            total_loss = 0

def evaluate(model, data_source, cuda=args.cuda):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    eval_loss = 0
    total_length = 0

    with torch.no_grad():
        for data_batch in data_source:
            data, target, length = process_data(data_batch, cuda=cuda, sep_target=sep_target)

            loss = model(data, target, length)
            cur_length = length.data.sum()
            eval_loss += loss.data[0] * cur_length
            total_length += cur_length

    return math.exp(eval_loss/total_length)


def run_epoch(epoch, lr, best_val_ppl):
    """A training epoch includes training, evaluation and logging"""
    epoch_start_time = time.time()
    train(model, corpus.train, epoch=epoch, lr=lr, weight_decay=args.weight_decay)
    val_ppl = evaluate(model, corpus.valid)
    logger.warning(
        '| end of epoch {:3d} | time: {:5.2f}s |'
        'valid ppl {:8.2f}'.format(
            epoch,
            (time.time() - epoch_start_time),
            val_ppl)
    )
    with open(args.save+'.epoch_{}'.format(epoch), 'wb') as f:
        torch.save(model, f)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_ppl or val_ppl < best_val_ppl:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_ppl = val_ppl
    else:
        # Anneal the learning rate if no improvement has been seen in the
        # validation dataset.
        lr /= args.lr_decay
    return lr, best_val_ppl

if __name__ == '__main__':
    lr = args.lr
    best_val_ppl = None
    basis_aneal_freq = args.epochs // 4
    if args.train:
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                lr, best_val_ppl = run_epoch(epoch, lr, best_val_ppl)
                if epoch % basis_aneal_freq == 0:
                    # reset the learning rate and revert basis mode
                    lr = args.lr
                    target_mode = not encoder.basis
                    logger.warning('Setting basis mode to {}'.format(target_mode))
                    encoder.basis_mode(target_mode)
        except KeyboardInterrupt:
            logger.warning('Exiting from training early')

    else:
        # Load the best saved model.
        with open(args.save, 'rb') as f:
            model = torch.load(f)

    # Run on test data.
    test_ppl = evaluate(model, corpus.test)
    logger.warning('| End of training | test ppl {:8.2f}'.format(test_ppl))
    sys.stdout.flush()
