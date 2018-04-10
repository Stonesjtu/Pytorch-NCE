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


world_size = distributed.get_world_size()
rank = distributed.get_rank()
sync_interval = 1

import time
def master(model, data_source, epoch, lr=1.0, weight_decay=1e-5, momentum=0.9):
    model = model.cuda()
    optimizer = optim.Adagrad(
        params=model.parameters(),
        weight_decay=weight_decay
    )

    def recv_grad(sender_rank):
        """Receive gradients computed by worker node specified by sender_rank"""
        counter = 0
        triplets = []
        while True:
            for name, p in model.named_parameters():
                tensor_buffer = p.new(p.size())
                event = distributed.irecv(tensor_buffer, sender_rank)
                triplets.append((name, p, tensor_buffer, event))

            while triplets:
                name, p, tensor_buffer, event = triplets.pop(0)
                # waiting for compeletion
                while not event.is_completed():
                    time.sleep(0.0005)
                if p.grad is None:
                    p.grad = tensor_buffer
                else:
                    p.grad.data.add_(tensor_buffer.data)
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
            if counter % sync_interval == 0:
                for p in model.parameters():
                    distributed.send(p.data, sender_rank)

    import threading
    threads = [threading.Thread(target=recv_grad, args=(i+1,)) for i in range(world_size - 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


count = 0
def worker(model, data_source, epoch, lr=1.0, weight_decay=1e-5, momentum=0.9):
    import time
    global count
    with torch.cuda.device(rank - 1):
        model = model.cuda()
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        # Turn on training mode which enables dropout.
        model.train()
        if rank == 1:
            pbar = tqdm(data_source, desc='Training PPL: ....')
            #pbar = data_source
        else:
            pbar = data_source

        events = []
        for num_batch, data_batch in enumerate(pbar):
            data, target, length = process_data(data_batch, cuda=args.cuda, sep_target=sep_target)
            loss = model(data, target, length)
            for e in events:
                e.wait()
            events = []
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

            torch.cuda.synchronize()
            for name, p in model.named_parameters():
                events.append(distributed.isend(p.grad.data, 0))

            count += 1
            if count % sync_interval == 0:
                for p in model.parameters():
                    tensor_buffer = p.new(p.size())
                    distributed.recv(tensor_buffer, 0)
                    p.data.set_(tensor_buffer.data)

def evaluate(model, data_source, cuda=args.cuda):
    # Turn on evaluation mode which disables dropout.
    with torch.cuda.device(rank - 1):
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
    else:
        master(model, corpus.train, epoch=epoch, lr=lr, weight_decay=args.weight_decay)
    val_ppl = evaluate(model, corpus.valid)
    logger.warn(
        'worker {} | end of epoch {:3d} | time: {:5.2f}s |'
        'valid ppl {:8.2f}'.format(
            rank,
            epoch,
            (time.time() - epoch_start_time),
            val_ppl)
    )
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
