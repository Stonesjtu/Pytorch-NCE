An NCE implementation in pytorch
===

About NCE
---

Noise Contrastive Estimation (NCE) is an approximation method that is used to work around
the huge computational cost of large softmax layer. The basic idea is to convert the prediction
problem into classification problem at training stage. It has been proved that these two criterions
converges to the same minimal point as long as noise distribution is close enough to real one.

NCE bridges the gap between generative models and discriminative models, rather than simply speedup
the softmax layer. With NCE, you can turn almost anything into posterior with less effort (I think).

Refs:

NCE:
> http://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann10AISTATS.pdf

NCE on rnnlm:
> https://pdfs.semanticscholar.org/144e/357b1339c27cce7a1e69f0899c21d8140c1f.pdf

### Comparison with other methods

A review of softmax speedup methods:
> http://ruder.io/word-embeddings-softmax/

NCE vs. IS (Importance Sampling): Nce is a binary classification while IS is sort of multi-class
classification problem.
> http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf

NCE vs. GAN (Generative Adversarial Network):
> https://arxiv.org/abs/1412.6515

### On improving NCE

#### Sampling methods

In NCE, unigram distribution is usually used to approximate the noise distribution because it's fast to
sample from. Sampling from a unigram is equal to multinomial sampling, which is of complexity $O(\log(N))$
via binary search tree. The cost of sampling becomes significant when noise ratio increases.

Since the unigram distribution can be obtained before training and remains unchanged across training,
some works are proposed to make use of this property to speedup the sampling procedure. Alias method is
one of them.

<img src="https://github.com/Stonesjtu/Pytorch-NCE/blob/master/res/alias.gif?raw=true" alt="diagram of constructing auxiliary data structure" height="200" />

By constructing data structures, alias method can reduce the sampling complexity from $O(log(N))$ to $O(1)$,
and it's easy to parallelize.

Refs:

alias method:
> https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

#### Generic NCE (full-NCE)

Conventional NCE only perform the contrasting on linear(softmax) layer, that is, given an input of a
linear layer, the model outputs are $p(noise|input)$ and $p(target|input)$. In fact NCE can be applied
to more general situations where models are capable to output likelihood values for both real data and
noise data.

In this code base, I use a variant of generic NCE named full-NCE (f-NCE) to clarify. Unlike normal NCE,
f-NCE samples the noises at input embedding.

Refs:

whole sentence language model by IBM (ICASSP2018)

Bi-LSTM language model by speechlab,SJTU (ICSLP2016?)

#### Batched NCE

Conventional NCE requires different noise samples per data token. Such computational pattern is not fully
GPU-efficient because it needs batched matrix multiplication. A trick is to share the noise samples across
the whole mini-batch, thus sparse batched matrix multiplication is converted to more efficient
dense matrix multiplication. The batched NCE is already supported by Tensorflow.

A more aggressive approach is to called self contrasting (named by myself). Instead of sampling from noise
distribution, the noises are simply the other training tokens the within the same mini-batch.

Ref:

batched NCE
> https://arxiv.org/pdf/1708.05997.pdf

self contrasting:
> https://www.isi.edu/natural-language/mt/simple-fast-noise.pdf

About the code
---

This NCE module is forked from the pytorch/examples repo.

### Requirements

Please run `pip install -r requirements` first to see if you have the required python lib.
- `tqdm` is used for process bar during training

### NCE related Arguments

- `--nce`: whether to use NCE as approximation
- `--noise-ratio <10>`: numbers of noise samples per data sample
- `--norm-term <9>`: the constant normalization term `Ln(z)`
- `--index-module <linear>`: index module to use for NCE module (currently
<linear> and <gru> available, <gru> does not support PPL calculating )
- `--train`: train or just evaluation existing model
- `--vocab <None>`: use vocabulary file if specified, otherwise use the words in train.txt
- `--loss [full, nce, sampled, mix]`: choose one of the loss type for training, the loss is
converted to `full` for PPL evaluation automatically.

### Examples

Run NCE criterion with linear module:
```bash
python main.py --cuda --noise-ratio 10 --norm-term 9 --nce --train
```

Run NCE criterion with gru module:
```bash
python main.py --cuda --noise-ratio 10 --norm-term 9 --nce --train --index-module gru
```

Run conventional CE criterion:
```bash
python main.py --cuda --train
```

### A small benchmark in swbd+fisher dataset

It's a performance showcase. The dataset is not bundled in this repo however.

#### dataset statistics
- training samples: 2200000 sentences, 22403872 words
- built vocabulary size: ~30K

#### testbed
- 1080 Ti
- i7 7700K
- pytorch-0.4.0
- cuda-8.0
- cudnn-6.0.1

#### how to run:
```bash
python main.py --train --batch-size 96 --cuda --loss nce --noise-ratio 500 --nhid 300 \
  --emsize 300 --log-interval 1000 --nlayers 1 --dropout 0 --weight-decay 1e-8 \
  --data data/swb --min-freq 3 --lr 2 --save nce-500-swb --concat
```

#### Running time
- crossentropy: 6.5 mins/epoch (56K tokens/sec)
- nce: 2 mins/epoch (187K tokens/sec)

#### performance

The rescore is performed on swbd 50-best, thanks to HexLee.

| training loss type | evaluation type | PPL     | WER                         |
| :---:              | :---:           | :--:    | :--:                        |
| CE                 | normed(full)    | 55      | 13.3                        |
| NCE                | unnormed(NCE)   | invalid | 13.4                        |
| NCE                | normed(full)    | 55      | 13.4                        |
| importance sample  | normed(full)    | 55      | 13.4                        |
| importance sample  | sampled(500)    | invalid | 19.0(worse than w/o rescore) |


### File structure

- `log/`: some log files of this scripts
- `alias_multinomial.py`: alias method sampling
- `nce.py`: the NCE module wrapper
- `vocab.py`: a wrapper for vocabulary object
- `index_linear.py`: an index module used by NCE, as a replacement for normal Linear module
- `index_gru.py`: an index module used by NCE, as a replacement for the whole language model module
- `model.py`: the wrapper of all `nn.Module`s.
- `generic_model.py`: the model wrapper for index_gru NCE module
- `main.py`: entry point
- `utils.py`: some util functions for better code structure

-----------------
### Modified README from Pytorch/examples

This example trains a multi-layer LSTM on a language modeling task.
By default, the training script uses the PTB dataset, provided.

```bash
python main.py --train --cuda --epochs 6        # Train a LSTM on PTB with CUDA
```

The model will automatically use the cuDNN backend if run on CUDA with
cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        humber of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --lr-decay         learning rate decay when no progress is observed on validation set
  --weight-decay     weight decay(L2 normalization)
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  --bptt             max length of truncated bptt
  --concat           use concatenated sentence instead of individual sentence
```
