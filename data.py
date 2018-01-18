"""data utils of this language model: corpus reader and noise data generator"""

import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from vocab import build_vocab

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


class PaddedDataset(Dataset):
    """dataset that zero-pads all sentence into same length
    Attributes:
        vocab_path: dictionary file, one word each line
        file_path: the directory of all train, test and valid corpus

    Parameters:
        dictionary: a word-to-index mapping, will build a new one if not provided
    """
    def __init__(self, file_path, vocab=None, vocab_path=None):
        super(PaddedDataset, self).__init__()
        self.file_path = file_path
        self.vocab_path = vocab_path
        self.file_path = file_path

        self.tokenize(file_path)
        if not vocab:
            self.vocab = build_vocab(file_path, min_freq=1)
        else:
            self.vocab = vocab

    def get_sentence_index(self, sentence):
        return [self.vocab.word2idx[word] for word in sentence]


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            sentences = []
            for line in tqdm(f, desc='Processing file: {}'.format(path)):
                sentence = ['<s>'] + line.split() + ['</s>']
                sentences.append(sentence)
        self.data = sentences

    def __getitem__(self, index):
        return self.get_sentence_index(self.data[index])

    def __len__(self):
        return len(self.data)

def pad_collate_fn(batch):
    """Pad the list of word indexes into 2-D LongTensor"""
    length = [len(sentence) for sentence in batch]
    return zero_padding(batch, length), torch.LongTensor(length)


class Corpus(object):
    def __init__(self, path, vocab_path=None, batch_size=1, shuffle=False, pin_memory=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'train.txt'), vocab_path=vocab_path)
        )
        self.vocab = self.train.dataset.vocab
        self.valid = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'valid.txt'), self.vocab)
        )
        self.test = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'test.txt'), self.vocab)
        )

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=pad_collate_fn,
            # waiting for a new torch version to support
            # drop_last=True,
        )

