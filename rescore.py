import math
import sys

import tqdm
import torch

from data import Corpus
from utils import process_data


MODEL_FILE = sys.argv[1]
# CORPUS_PATH = '../language_model/data/swb-bpe-dp-extreme'
CORPUS_PATH = '../language_model/data/swb-rescore'
VOCAB_PATH = '../language_model/data/swb-bpe-tri/vocab.txt'
#################################################################
# Load data
#################################################################
corpus = Corpus(
    path=CORPUS_PATH,
    # vocab_path=VOCAB_PATH,
    min_freq=3,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    # concat=True,
    bptt=120,
)


model = torch.load(MODEL_FILE)

print('vocabulary size: ', len(corpus.vocab.idx2word))
print('sample words: ', corpus.vocab.idx2word[:10])


data_source = corpus.test
# Turn on evaluation mode which disables dropout.
model.eval()
model.criterion.loss_type = 'nce'
model.criterion.noise_ratio = 500
print('Rescoring using loss: {}'.format(model.criterion.loss_type))

# GRU does not support ce mode right now
eval_loss = 0
total_length = 0

scores = []

debug = False

with torch.no_grad():
    for data_batch in tqdm.tqdm(data_source):
        data, target, length = process_data(data_batch, cuda=False, sep_target=False)

        if debug:
            print(model(data, length.cuda()))
            continue
        loss = model(data, length.cuda()).item()
        loss *= length.sum().item()
        eval_loss += loss
        total_length += length.sum().item()
        score = - loss / math.log(2)  # change the base from e to 2
        scores.append('{:.8f}'.format(score))

print('PPL: ', math.exp(eval_loss / total_length))
with open('./score.txt', 'w') as f_out:
    f_out.write('\n'.join(scores))
