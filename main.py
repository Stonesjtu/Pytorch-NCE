#!/usr/bin/env python

import sys
import time
import math

from tqdm import tqdm

import torch
from torch import nn, optim, autograd, distributed

import data
from model import RNNModel
from utils import process_data, setup_parser, setup_logger


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
distributed.init_process_group(backend='mpi', world_size=0, rank=0)
corpus = data.Corpus(
    path=args.data,
    vocab_path=args.vocab,
    batch_size=args.batch_size,
    min_freq=args.min_freq,
)

ntoken = len(corpus.train.dataset.vocab)
sep_target = True
################################################################## Build the criterion and model, setup the NCE module
#################################################################

model = RNNModel(ntoken=ntoken, ninp=args.emsize, nhid=args.nhid,
                 nlayers=args.nlayers, dropout=args.dropout)

ntoken = len(corpus.train.dataset.vocab)
logger.info('Vocabulary size is {}'.format(ntoken))

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
    total_loss = 0
    pbar = tqdm(data_source, desc='Training PPL: ....')
    for num_batch, data_batch in enumerate(pbar):
        optimizer.zero_grad()
        data, target, length = process_data(data_batch, cuda=args.cuda, sep_target=sep_target)
        loss = model(data, target, length)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data[0]

        if args.prof:
            break
        if num_batch % args.log_interval == 0 and num_batch > 0:
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

world_size = distributed.get_world_size()
rank = distributed.get_rank()
sync_interval = 200

def master(model, data_source, epoch, lr=1.0, weight_decay=1e-5, momentum=0.9):
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    model = model.cuda()

    def recv_grad(sender_rank):
        """Recive gradients computed by worker node specified by sender_rank"""
        counter = 0
        while True:
            for p in model.parameters():
                tensor_buffer = torch.Tensor(p.size())
                distributed.recv(tensor_buffer, sender_rank)
                if p.grad is not None:
                    p.grad.data += tensor_buffer.cuda()
                else:
                    p.grad = tensor_buffer.cuda()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
            if counter == sync_interval:
                counter = 0
                print('syncing parameters')
                for p in model.parameters():
                    distributed.isend(p.data.cpu(), sender_rank)

    import threading
    threads = [threading.Thread(target=recv_grad, args=(i+1,)) for i in range(world_size - 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def worker(model, data_source, epoch, lr=1.0, weight_decay=1e-5, momentum=0.9):
    with torch.cuda.device(rank):
        model = model.cuda()
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        if rank == 1:
            pbar = tqdm(data_source, desc='Training PPL: ....')
        else:
            pbar = data_source
        for num_batch, data_batch in enumerate(pbar):
            data, target, length = process_data(data_batch, cuda=args.cuda, sep_target=sep_target)
            loss = model(data, target, length)
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            for p in model.parameters():
                # p.grad.data.zero_()
                distributed.isend(p.grad.data.cpu(), 0)
                optimizer.step()

            total_loss += loss.data[0]
            if (num_batch+1) % sync_interval == 0 and num_batch > 0:

                for p in model.parameters():
                    tensor_buffer = torch.Tensor(p.size())
                    distributed.recv(tensor_buffer, 0)
                    p.data.set_(tensor_buffer.cuda())

                if rank == 1:
                    cur_loss = total_loss / sync_interval #args.log_interval
                    ppl = math.exp(cur_loss)
                    pbar.set_description('Training PPL %.1f' % ppl)
                    total_loss = 0

def evaluate(model, data_source, cuda=args.cuda):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    eval_loss = 0
    total_length = 0

    for data_batch in data_source:
        data, target, length = process_data(data_batch, cuda=cuda, sep_target=sep_target)

        loss = model(data, target, length)
        cur_length = int(length.data.sum())
        eval_loss += loss.data[0] * cur_length
        total_length += cur_length

    return math.exp(eval_loss/total_length)


def run_epoch(epoch, lr, best_val_ppl):
    """A training epoch includes training, evaluation and logging"""
    epoch_start_time = time.time()
    if rank != 0:
        worker(model, corpus.train, epoch=epoch, lr=lr, weight_decay=args.weight_decay)
        return lr, best_val_ppl
    master(model, corpus.train, epoch=epoch, lr=lr, weight_decay=args.weight_decay)
    val_ppl = evaluate(model, corpus.valid)
    logger.warn(
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
    if args.train:
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                lr, best_val_ppl = run_epoch(epoch, lr, best_val_ppl)
                if args.prof:
                    break
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
