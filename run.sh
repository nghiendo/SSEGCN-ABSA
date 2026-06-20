#!/bin/bash
# Usage:
#   bash ./run.sh
#
# Notes:
# - The plain `ssegcn` model does not consume KG3/Node2Vec knowledge features.
# - To use knowledge-enhanced features, run a `ssegcn_bert` command with
#   `--use_node2vec` and `--node2vec_file`.
# - Before the knowledge-enhanced run, make sure KG3 files exist:
#     python dataset/preprocess_kg3.py --dataset laptop
# - And make sure the vocab files are prepared:
#     python prepare_vocab.py --data_dir dataset/Laptops_corenlp --vocab_dir dataset/Laptops_corenlp
#
# Replace `path/to/node2vec_embeddings.json` below with your actual embedding file.

# * laptop

# * SSEGCN
python ./train.py --model_name ssegcn --dataset laptop --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Laptops_corenlp --cuda 0  

# * SSEGCN with DeBERTa
# python ./train.py --model_name ssegcn_bert --dataset laptop --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 100 --cuda 0  

# * SSEGCN with DeBERTa + KG3/Node2Vec knowledge
# python ./train.py --model_name ssegcn_bert --dataset laptop --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 10 --hidden_dim 768 --max_length 100 --use_node2vec --node2vec_dim 128 --node2vec_file path/to/node2vec_embeddings.json --cuda 0


# * restaurant

# * SSEGCN
# python ./train.py --model_name ssegcn --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --cuda 0  
# * SSEGCN with DeBERTa
# python ./train.py --model_name ssegcn_bert --dataset restaurant --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 
# * SSEGCN with DeBERTa + KG3/Node2Vec knowledge
# python ./train.py --model_name ssegcn_bert --dataset restaurant --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --use_node2vec --node2vec_dim 128 --node2vec_file path/to/node2vec_embeddings.json --cuda 0

# * twitter

# * SSEGCN
# python ./train.py --model_name ssegcn --dataset twitter --seed 1000 --num_epoch 40 --vocab_dir ./dataset/Tweets_corenlp --cuda 0 

# * SSEGCN with DeBERTa
# python ./train.py --model_name ssegcn_bert --dataset twitter --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 
# * SSEGCN with DeBERTa + KG3/Node2Vec knowledge
# python ./train.py --model_name ssegcn_bert --dataset twitter --seed 1000 --pretrained_bert_name microsoft/deberta-base --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --use_node2vec --node2vec_dim 128 --node2vec_file path/to/node2vec_embeddings.json --cuda 0
