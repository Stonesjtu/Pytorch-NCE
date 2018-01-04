This NCE module is forked from the pytorch/examples repo.

### Requirements

Please run `pip install -r requirements` first to see if you have the required python lib.
- `tqdm` is used for process bar during training

### New Arguments

- `--nce`: whether to use NCE as approximation
- `--noise-ratio <10>`: numbers of noise samples per data sample
- `--norm-term <9>`: the constant normalization term `Ln(z)`
- `--index-module <linear>`: index module to use for NCE module (currently
<linear> and <gru> available, <gru> does not support PPL calculating )
- `--train`: train or just evaluation existing model
- `--vocab <None>`: use vocabulary file if specified, otherwise use the words in train.txt

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

### File structure

- `log/`: some log files of this scripts
- `nce.py`: the NCE module wrapper
- `index_linear.py`: an index module used by NCE, as a replacement for normal Linear module
- `index_gru.py`: an index module used by NCE, as a replacement for the whole language model module
- `model.py`: the wrapper of all `nn.Module`s.
- `main.py`: entry point
- `utils.py`: some util functions for better abstraction

-----------------
### Modified README from Pytorch/examples

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6        # Train a LSTM on PTB with CUDA
```

The model uses the `nn.LSTM` module which will automatically use the cuDNN backend if run on CUDA with
cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluted against the test dataset.

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
```
