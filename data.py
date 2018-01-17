"""data utils of this language model: corpus reader and noise data generator"""

import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import tqdm


def zero_padding(sentences, length):
    """
    sentences: a list of sentence
    length: the valid length of corresponding sentences
    """
    max_len = max(length)
    padded_sentences = []
    for length, sentence in zip(length, sentences):
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
        self.idx2count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.idx2count.append(0)

    def incre_count(self, idx):
        self.idx2count[idx] += 1

    def trunc_special(self):
        """Do not count special characters as `<s>` """
        special_words = ['<s>']
        for word in special_words:
            idx = self.word2idx[word]
            self.idx2count[idx] = 0

    def __len__(self):
        return len(self.idx2word)


class PaddedDataset(Dataset):
    """dataset that zero-pads all sentence into same length
    Attributes:
        vocab_path: dictionary file, one word each line
        file_path: the directory of all train, test and valid corpus

    Parameters:
        dictionary: a word-to-index mapping, will build a new one if not provided
    """
    def __init__(self, file_path, dictionary=None, vocab_path=None):
        super(PaddedDataset, self).__init__()
        self.file_path = file_path
        self.vocab_path = vocab_path

        self.dictionary = Dictionary()
        if not dictionary:
            self._build_dict()
        else:
            self.dictionary = dictionary
        self.file_path = file_path
        self.data = self.tokenize(file_path)

    def _build_dict(self):
        """build the dictionary before the training phase

        If dictionary file is provided, then use it directly.
        Otherwise use every words in train corpus.
        """

        # Use existing vocabulary file to construct dict
        if self.vocab_path:
            assert os.path.exists(self.vocab_path)
            # Add words to the dictionary
            with open(self.vocab_path, 'r') as f:
                for line in f:
                    self.dictionary.add_word(line.split()[0])


        # Use train corpus
        else:
            assert os.path.exists(self.file_path)
            # Add words to the dictionary
            with open(self.file_path, 'r') as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        self.dictionary.add_word(word)

        # Ensure the special characters are in vocabulary
        self.dictionary.add_word('<s>')
        self.dictionary.add_word('</s>')
        self.dictionary.add_word('<unk>')


    def get_index(self, word):
        """Get indices in vocabulary

        At the same time, this function will increase the word count by 1
        """
        if word in self.dictionary.word2idx:
            idx = self.dictionary.word2idx[word]
        else:
            idx = self.dictionary.word2idx['<unk>']

        self.dictionary.incre_count(idx)
        return idx


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            sentences = []
            for line in tqdm.tqdm(f):
                words = ['<s>'] + line.split() + ['</s>']
                sentence = torch.LongTensor(
                    [self.get_index(word) for word in words])
                sentences.append(sentence)
        self.dictionary.trunc_special()
        return sentences

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def pad_collate_fn(batch):
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
            collate_fn=pad_collate_fn,
            # waiting for a new torch version to support
            # drop_last=True,
        )

