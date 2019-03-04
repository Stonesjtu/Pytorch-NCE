#!/usr/bin/env bash

set -x

ipdb3 main.py --batch-size 128 --cuda --loss sampled --noise-ratio 5000 --nhid 2048 \
           --emsize 256 --log-interval 1000 --nlayers 1 --dropout 0.01 --weight-decay 1e-6 \
           --data ../dataset/obw --min-freq 3 --lr 0.001 --save nce-5000-obw-is-test --concat --bptt 20 --clip 1
