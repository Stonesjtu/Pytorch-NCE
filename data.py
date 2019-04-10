"""data utils of this language model: corpus reader and noise data generator"""

import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from vocab import get_vocab, BOS, EOS


class LMDataset(Dataset):
    """dataset that zero-pads all sentence into same length

    Attributes:
        - vocab: Vocab object which holds the vocabulary info
        - file_path: the directory of all train, test and valid corpus
        - bptt: truncated BPTT length, items after such length will be
        ignored

    Parameters:
        - vocab: a word-to-index mapping, will build a new one if
        not provided
    """

    def __init__(self, file_path, vocab=None, bptt=35):
        super(LMDataset, self).__init__()
        self.vocab = vocab
        self.file_path = file_path
        self.bptt = bptt
        self.tokenize(file_path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            sentences = []
            for sentence in tqdm(f, desc='Processing file: {}'.format(path)):
                sentences.append(sentence.split())
        self.data = sentences

    def __getitem__(self, index):
        raw_sentence = self.data[index]
        # truncate the sequence length to maximum of BPTT
        sentence = [BOS] + raw_sentence[:self.bptt] + [BOS]
        return [self.vocab.word2idx[word] for word in sentence]

    def __len__(self):
        return len(self.data)


class ContLMDataset(LMDataset):
    """dataset that cat the sentences into one long sequence and chunk

    Each training sample is a chunked version and of same length.

    Attributes:
        - vocab: Vocab object which holds the vocabulary info
        - file_path: the directory of all train, test and valid corpus
        - bptt: sequence length
    """

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # add the start of sentence token
        sentence_sep = [BOS]
        with open(path, 'r') as f:
            sentences = [BOS]
            for sentence in tqdm(f, desc='Processing file: {}'.format(path)):
                sentences += sentence.split() + sentence_sep
        # split into list of tokens
        self.data = sentences

    def __getitem__(self, index):
        sentence = self.data[index * self.bptt:(index + 1) * self.bptt]
        return [self.vocab.word2idx[word] for word in sentence]

    def __len__(self):
        return len(self.data) // self.bptt


def pad_collate_fn(batch):
    """Pad the list of word indexes into 2-D LongTensor"""
    length = [len(sentence) for sentence in batch]
    return pad_sequence([torch.LongTensor(s) for s in batch], batch_first=True), torch.LongTensor(length)


class Corpus(object):
    def __init__(self, path, vocab_path=None, batch_size=1, shuffle=False,
                 pin_memory=False, update_vocab=False, min_freq=1,
                 concat=False, bptt=35):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.base_path = path
        self.update_vocab = update_vocab
        self.bptt = bptt
        self.concat = concat

        self.vocab = get_vocab(path, ['train.txt'], min_freq=min_freq, vocab_file=vocab_path)
        if self.concat:
            # set the frequencies for special tokens by miracle trial
            self.vocab.idx2count[1] = self.vocab.freqs[BOS]  # <s>
            self.vocab.idx2count[2] = 0  # </s>

        self.train = self.get_dataloader('train.txt', self.batch_size)
        self.valid = self.get_dataloader('valid.txt', 1)
        self.test = self.get_dataloader('test.txt', 1)

    def get_dataloader(self, filename, bs=1):
        full_path = os.path.join(self.base_path, filename)
        if self.concat:
            dataset = ContLMDataset(full_path, vocab=self.vocab, bptt=self.bptt)
        else:
            dataset = LMDataset(full_path, vocab=self.vocab, bptt=self.bptt)
        return DataLoader(
            dataset=dataset,
            batch_size=bs,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=pad_collate_fn,
            # num_workers=1,
            # waiting for a new torch version to support
            drop_last=True,
        )
