#!/bin/bash
set -euo pipefail

python3 ./train.py \
  --model_name ssegcnbert \
  --dataset twitter \
  --seed 1000 \
  --num_epoch 1 \
  --batch_size 16 \
  --log_step 10 \
  --max_length 100 \
  --hidden_dim 768 \
  --bert_lr 2e-5 \
  --learning_rate 2e-3 \
  --cuda 0
