"""data utils of this language model: corpus reader and noise data generator"""

import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from vocab import get_vocab

def zero_padding(sentences, length):
    """
    sentences: a list of sentence
    length: the valid length of corresponding sentences
    """
    max_len = max(length)
    padded_sentences = []
    for length, sentence in zip(length, sentences):
        padding_length = max_len - length
        sentence = torch.LongTensor(sentence)
        padding = torch.LongTensor(padding_length).zero_()
        padded_sentence = torch.cat((sentence, padding), 0)
        padded_sentences.append(padded_sentence)
    padded_sentences = torch.stack(padded_sentences)
    return padded_sentences


class LMDataset(Dataset):
    """dataset that zero-pads all sentence into same length

    Attributes:
        - vocab_path: dictionary file, one word each line
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
                sentences.append(sentence)
        self.data = sentences

    def __getitem__(self, index):
        raw_sentence = self.data[index]
        # truncate the sequence length to maximum of BPTT
        sentence = ['<s>'] + raw_sentence.split()[:self.bptt] + ['</s>']
        return [self.vocab.word2idx[word] for word in sentence]

    def __len__(self):
        return len(self.data)


class ContLMDataset(LMDataset):
    """dataset that cat the sentences into one long sequence and chunk

    Each training sample is a chunked version and of same length.

    Attributes:
        - vocab_path: dictionary file, one word each line
        - file_path: the directory of all train, test and valid corpus
        - bptt: sequence length
    """

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            sentences = []
            for sentence in tqdm(f, desc='Processing file: {}'.format(path)):
                sentences.append(sentence)

        # add the end of sentence token
        concated_seq = ' </s> '.join(sentences)
        # split into list of tokens
        self.data = concated_seq.split()

    def __getitem__(self, index):
        sentence = self.data[index * self.bptt:(index + 1) * self.bptt]
        return [self.vocab.word2idx[word] for word in sentence]

    def __len__(self):
        return len(self.data) // self.bptt


def pad_collate_fn(batch):
    """Pad the list of word indexes into 2-D LongTensor"""
    length = [len(sentence) for sentence in batch]
    return zero_padding(batch, length), torch.LongTensor(length)


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

        self.vocab = get_vocab(path, ['train.txt'], min_freq=min_freq)
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
            # drop_last=True,
        )
