#!/bin/bash

model_path=$1

PYTHONPATH=. python grolp/eval/eval_seq2seq.py --model_path $model_path --eval_split valid_seen --cuda_device 0 --num_workers 4;
PYTHONPATH=. python grolp/eval/eval_seq2seq.py --model_path $model_path --eval_split valid_unseen --cuda_device 0 --num_workers 4;