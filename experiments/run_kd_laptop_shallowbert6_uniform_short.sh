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
  --student_init_path ./state_dict/ssegcnbert_laptop_acc_0.7722_f1_0.7031 \
  --student_bert_layers 6 \
  --student_bert_layer_map uniform \
  --student_bert_use_adamw true \
  --bert_lr 2e-5 \
  --learning_rate 2e-4 \
  --teacher_path ./state_dict/ssegcnbert_laptop_acc_0.7722_f1_0.7031 \
  --teacher_model_name ssegcnbert \
  --kd_temperature 2.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.5 \
  --kd_beta 0.35 \
  --kd_gamma 0.15 \
  --kd_feature_loss cosine \
  --kd_use_instance_weighting false \
  --cuda 0
