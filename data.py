#  data utils of this language model: corpus reader and noise data generator
import os
from nltk.probability import FreqDist
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def zero_padding(sentences, lengths):
    """
    sentences: a list of sentence
    lengths: the valid lengths of corresponding sentences
    """
    max_len = max(lengths)
    padded_sentences = []
    for length, sentence in zip(lengths, sentences):
        padding_length = max_len - length
        padding = torch.LongTensor(padding_length).zero_()
        padded_sentence = torch.cat((sentence, padding), 0)
        padded_sentences.append(padded_sentence)
    padded_sentences = torch.stack(padded_sentences)
    return padded_sentences


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class PaddedDataset(Dataset):
    """dataset that zero-pads all sentence into same length
    Parameters:
        file_path: the directory of all train, test and valid corpus
        dictionary: a word-to-index mapping, will build a new one if not provided
    """
    def __init__(self, file_path, dictionary=None):
        super(PaddedDataset, self).__init__()

        self.dictionary = Dictionary()
        if not dictionary:
            self._build_dict(file_path)
        else:
            self.dictionary = dictionary
        self.file_path = file_path
        self.data, self.lengths = self.tokenize(file_path)

    def _build_dict(self, path):
        """build the dictionary before the training phase
        Parameters:
            path: training corpus location
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            sentences = []
            lengths = []
            for line in f:
                words = line.split() + ['<eos>']
                lengths.append(len(words))
                sentence = torch.LongTensor(
                    [self.dictionary.word2idx[word] for word in words])
                sentences.append(sentence)
        lengths = torch.ShortTensor(lengths)
        padded_sentences = zero_padding(sentences, lengths)
        return padded_sentences, lengths

    def __getitem__(self, index):
        return (
            self.data[index][:-1],
            self.data[index][1:],
            self.lengths[index] - 1,
        )

    def __len__(self):
        return len(self.data)


class Corpus(object):
    def __init__(self, path, batch_size=1, shuffle=False, pin_memory=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'train.txt'))
        )
        self.dict = self.train.dataset.dictionary
        self.valid = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'valid.txt'), self.dict)
        )
        self.test = self.get_dataloader(
            PaddedDataset(os.path.join(path, 'test.txt'), self.dict)
        )

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            # waiting for a new torch version to support
            # drop_last=True,
        )

