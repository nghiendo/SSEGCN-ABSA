#!/bin/bash
set -euo pipefail

python ./train.py \
  --model_name ssegcnbertstudent \
  --dataset laptop \
  --seed 1000 \
  --num_epoch 1 \
  --batch_size 16 \
  --log_step 10 \
  --max_length 100 \
  --vocab_dir ./dataset/Laptops_corenlp \
  --student_hidden_dim 40 \
  --student_pos_dim 8 \
  --student_post_dim 8 \
  --student_lr 3e-5 \
  --student_freeze_word_emb false \
  --student_init_path ./state_dict/ssegcnbertstudent_laptop_acc_0.6851_f1_0.6350 \
  --student_expand_init true \
  --teacher_student_hidden_dim 32 \
  --teacher_student_pos_dim 8 \
  --teacher_student_post_dim 8 \
  --teacher_student_encoder_layers 1 \
  --aux_teacher_path ./state_dict/ssegcnbertstudent_laptop_acc_0.6851_f1_0.6350 \
  --aux_teacher_model_name ssegcnbertstudent \
  --aux_teacher_logit_weight 0.2 \
  --aux_teacher_blend_mode confidence \
  --aux_teacher_blend_space probs \
  --aux_teacher_gate_temperature 0.05 \
  --kd_temperature 4.0 \
  --kd_blended_target_temperature 4.0 \
  --kd_alpha 0.65 \
  --kd_beta 0.35 \
  --kd_gamma 0.0 \
  --kd_feature_loss cosine \
  --cuda 0
