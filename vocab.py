"""Build the vocabulary from corpus

This file is forked from pytorch/text repo at Github.com"""
import os
import pickle
import logging
from collections import defaultdict, Counter

from tqdm import tqdm
logger = logging.getLogger(__name__)


def _default_unk_index():
    return 0

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        word2idx: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        idx2word: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
        """
        self.freqs = counter
        self.max_size = max_size
        self.min_freq= min_freq
        self.specials = specials
        self.vectors = vectors
        self.build()


    def build(self):
        """Build the required vocabulary according to attributes"""
        counter = self.freqs.copy()
        min_freq = max(self.min_freq, 1)
        counter.update(self.specials)

        self.idx2word = list(self.specials)
        self.idx2count = []

        counter.subtract({tok: counter[tok] for tok in self.specials})
        max_size = None if self.max_size is None else self.max_size + len(self.idx2word)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.idx2word) == max_size:
                break
            self.idx2word.append(word)
            self.idx2count.append(freq)
        self.word2idx = {
            word: idx for idx, word in enumerate(self.idx2word)
        }
        self.word2idx = defaultdict(_default_unk_index)
        self.word2idx.update({tok: i for i, tok in enumerate(self.specials)})

    def save(self, filename):
        """Save the counter of vocabulary for speed performance

        The counter is the most time-consuming object to obtain,
        so we only save the counter for convinient
        """
        pickle.dump(self.freqs, open(filename, 'wb'))

    def load(self, filename):
        self.freqs = pickle.load(open(filename, 'rb'))

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.word2idx != other.word2idx:
            return False
        if self.idx2word != other.idx2word:
            return False
        return True

    def __len__(self):
        return len(self.idx2word)

    def extend(self, v, sort=False):
        words = sorted(v.idx2word) if sort else v.idx2word
        #TODO: speedup the dependency
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.idx2word) - 1

    def write_txt(self, filename):
        """Write the vocabulary into text file"""
        write_str = ['{} {}'.format(pair) for pair in zip(self.idx2word, self.idx2count)]
        with open(filename, 'w') as f:
            f.writelines(write_str)


def build_vocab(filename, min_freq, force_recount=False):
    """Build vocabulary file with each line the word and frequency

    Args:
        - sentences: sentences with BOS and EOS
        - min_freq: minimal frequency to truncate
        - force_recount: force a re-count of word frequency regardless of the
        Count cache file

    Return:
        - vocab: the Vocab object
    """
    counter = Counter()
    cache_file = filename+'.Vocab'
    if os.path.exists(cache_file) and not force_recount:
        logger.info('Load cached vocabulary object')
        vocab = pickle.load(open(cache_file, 'rb'))
        vocab.min_freq = min_freq
        vocab.build()
        logger.info('Load cached vocabulary object finished')
    else:
        logger.info('Refreshing vocabulary')
        for line in tqdm(open(filename, 'r'), desc='Building vocabulary: '):
            counter.update(line.split())
            counter.update(['<s>', '</s>'])
        vocab = Vocab(counter, min_freq=min_freq, specials=[])
        logger.info('Refreshing vocabulary finished')
        pickle.dump(vocab, open(cache_file, 'wb'))
    return vocab
