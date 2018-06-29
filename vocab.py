"""Build the vocabulary from corpus

This file is forked from pytorch/text repo at Github.com"""
import os
import dill as pickle
import logging
from collections import defaultdict, Counter

from tqdm import tqdm
logger = logging.getLogger(__name__)


def _default_unk_index():
    return 0


def load_freq(freq_file):
    """Load the frequency from text file"""
    counter = Counter()
    with open(freq_file) as f:
        for line in f:
            word, freq = line.split(' ')
            counter[word] = freq
    return counter


def write_freq(counter, freq_file):
    """Write the word-frequency pairs into text file

    File format:

        word1 freq1
        word2 freq2

    """
    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    with open(freq_file, 'w') as f:
        for word, freq in words_and_frequencies:
            f.writelines('{} {}\n'.format(word, freq))



class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        word2idx: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        idx2word: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, counter, max_size=None, min_freq=1):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
        """
        self.freqs = counter
        self.max_size = max_size
        self.min_freq = min_freq
        self.specials = ['<unk>', '<s>']
        self.build()


    def build(self, force_vocab=[]):
        """Build the required vocabulary according to attributes

        We need an explicit <unk> for NCE because this improve the precision of
        word frequency estimation in noise sampling

        Args:
            - force_vocab: force the vocabulary to be within this vocab
        """
        counter = self.freqs.copy()
        if force_vocab:
            min_freq = 1
        min_freq = max(self.min_freq, 1)

        # delete the special tokens from given vocabulary
        force_vocab = [word for word in force_vocab if word not in self.specials] + ['</s>']
        self.idx2word = list(self.specials) + force_vocab

        # Do not count the BOS and UNK as frequency term
        for word in self.specials:
            del counter[word]
        max_size = None if self.max_size is None else self.max_size + len(self.idx2word)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        unk_freq = 0
        for word, freq in words_and_frequencies:

            # for words not in force_vocab and with freq<th, throw to <unk>
            if freq < min_freq and word not in force_vocab:
                unk_freq += freq
            elif len(self.idx2word) != max_size and not force_vocab:
                self.idx2word.append(word)

        self.word2idx = defaultdict(_default_unk_index)
        self.word2idx.update({
            word: idx for idx, word in enumerate(self.idx2word)
        })

        self.idx2count = [self.freqs[word] for word in self.idx2word]
        self.idx2count[0] += unk_freq
        self.idx2count[1] = 0

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
        # TODO: speedup the dependency
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.idx2word) - 1


def check_vocab(vocab):
    """A util function to check the vocabulary correctness"""
    # one word for one index
    assert len(vocab.idx2word) == len(vocab.word2idx)

    # no duplicate words in idx2word
    assert len(set(vocab.idx2word)) == len(vocab.idx2word)


def get_vocab(base_path, file_list, min_freq=1, force_recount=False, vocab_file=None):
    """Build vocabulary file with each line the word and frequency

    The vocabulary object is cached at the first build, aiming at reducing
    the time cost for pre-process during training large dataset

    Args:
        - sentences: sentences with BOS and EOS
        - min_freq: minimal frequency to truncate
        - force_recount: force a re-count of word frequency regardless of the
        Count cache file
        - vocab_file: a specific vocabulary file. If not None, the returned
        vocabulary will only count the words in vocab_file, with others treated
        as <unk>

    Return:
        - vocab: the Vocab object
    """
    counter = Counter()
    cache_file = os.path.join(base_path, 'vocab.pkl')

    if os.path.exists(cache_file) and not force_recount:
        logger.debug('Load cached vocabulary object')
        vocab = pickle.load(open(cache_file, 'rb'))
        if min_freq:
            vocab.min_freq = min_freq
        logger.debug('Load cached vocabulary object finished')
    else:
        logger.debug('Refreshing vocabulary')
        for filename in file_list:
            full_path = os.path.join(base_path, filename)
            for line in tqdm(open(full_path, 'r'), desc='Building vocabulary: '):
                counter.update(line.split())
                counter.update(['<s>', '</s>'])
        vocab = Vocab(counter, min_freq=min_freq)
        logger.debug('Refreshing vocabulary finished')

        # saving for future uses
        freq_file = os.path.join(base_path, 'freq.txt')
        write_freq(vocab.freqs, freq_file)
        pickle.dump(vocab, open(cache_file, 'wb'))

    force_vocab = []
    if vocab_file:
        with open(vocab_file) as f:
            force_vocab = set([line.strip() for line in f])
    vocab.build(force_vocab=force_vocab)
    check_vocab(vocab)
    return vocab
