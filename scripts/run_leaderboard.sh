#!/bin/bash

model_path=$1

PYTHONPATH=. python grolp/eval/leaderboard.py --model_path $model_path --eval_split tests_seen --cuda_device 0 --num_workers 4;
PYTHONPATH=. python grolp/eval/leaderboard.py --model_path $model_path --eval_split tests_unseen --cuda_device 0 --num_workers 4;