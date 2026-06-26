#!/bin/bash
set -euo pipefail

python ./train.py \
  --model_name ssegcnbertshallow \
  --dataset laptop \
  --seed 1000 \
  --num_epoch 2 \
  --batch_size 16 \
  --log_step 10 \
  --max_length 100 \
  --student_init_path ./state_dict/ssegcnbertshallow_laptop_acc_0.7658_f1_0.7275 \
  --student_bert_layers 6 \
  --student_bert_layer_map last \
  --student_bert_use_adamw true \
  --bert_lr 5e-6 \
  --learning_rate 2e-4 \
  --teacher_path ./state_dict/ssegcnbert_laptop_acc_0.7722_f1_0.7031 \
  --teacher_model_name ssegcnbert \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.70 \
  --kd_beta 0.25 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.02 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false \
  --cuda 0
