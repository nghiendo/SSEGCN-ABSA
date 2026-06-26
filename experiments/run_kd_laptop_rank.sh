#!/bin/bash
set -euo pipefail

python ./train.py \
  --model_name ssegcnbertstudent \
  --dataset laptop \
  --seed 1000 \
  --num_epoch 10 \
  --batch_size 16 \
  --log_step 10 \
  --max_length 100 \
  --vocab_dir ./dataset/Laptops_corenlp \
  --student_hidden_dim 32 \
  --student_pos_dim 8 \
  --student_post_dim 8 \
  --student_lr 5e-4 \
  --student_freeze_word_emb false \
  --kd_temperature 4.0 \
  --kd_alpha 0.4 \
  --kd_beta 0.4 \
  --kd_gamma 0.2 \
  --kd_rank_weight 0.01 \
  --kd_rank_temperature 1.0 \
  --cuda 0
