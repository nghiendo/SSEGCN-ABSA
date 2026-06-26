#!/bin/bash
set -euo pipefail

python ./train.py \
  --model_name ssegcnbertstudent \
  --dataset laptop \
  --seed 1000 \
  --num_epoch 4 \
  --batch_size 16 \
  --log_step 10 \
  --max_length 100 \
  --vocab_dir ./dataset/Laptops_corenlp \
  --student_hidden_dim 32 \
  --student_pos_dim 8 \
  --student_post_dim 8 \
  --student_lr 2e-4 \
  --student_freeze_word_emb false \
  --student_init_path ./state_dict/ssegcnbertstudent_laptop_acc_0.6883_f1_0.6325 \
  --kd_temperature 4.0 \
  --kd_alpha 0.5 \
  --kd_beta 0.35 \
  --kd_gamma 0.15 \
  --kd_feature_loss cosine \
  --kd_token_relation_weight 0.01 \
  --cuda 0
