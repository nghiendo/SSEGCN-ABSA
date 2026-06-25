#!/bin/bash

# * laptop

# * Teacher: SSEGCN-BERT
python ./train.py --model_name ssegcnbert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 100 --vocab_dir ./dataset/Laptops_corenlp --cuda 0

# * Student: distill from SSEGCN-BERT
# By default, student auto-loads the latest teacher checkpoint from ./state_dict for the same dataset.
# python ./train.py --model_name ssegcnbertstudent --dataset laptop --seed 1000 --num_epoch 20 --max_length 100 --vocab_dir ./dataset/Laptops_corenlp --student_hidden_dim 32 --student_pos_dim 8 --student_post_dim 8 --student_lr 1e-3 --kd_temperature 4.0 --kd_alpha 0.4 --kd_beta 0.4 --kd_gamma 0.2 --cuda 0


# * restaurant

# * Teacher: SSEGCN-BERT
# python ./train.py --model_name ssegcnbert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0

# * Student: distill from SSEGCN-BERT
# python ./train.py --model_name ssegcnbertstudent --dataset restaurant --seed 1000 --num_epoch 20 --max_length 100 --vocab_dir ./dataset/Restaurants_corenlp --student_hidden_dim 32 --student_pos_dim 8 --student_post_dim 8 --student_lr 1e-3 --kd_temperature 4.0 --kd_alpha 0.4 --kd_beta 0.4 --kd_gamma 0.2 --cuda 0


# * twitter

# * Teacher: SSEGCN-BERT
# python ./train.py --model_name ssegcnbert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --vocab_dir ./dataset/Tweets_corenlp --cuda 0

# * Student: distill from SSEGCN-BERT
# python ./train.py --model_name ssegcnbertstudent --dataset twitter --seed 1000 --num_epoch 20 --max_length 100 --vocab_dir ./dataset/Tweets_corenlp --student_hidden_dim 32 --student_pos_dim 8 --student_post_dim 8 --student_lr 1e-3 --kd_temperature 4.0 --kd_alpha 0.4 --kd_beta 0.4 --kd_gamma 0.2 --cuda 0
