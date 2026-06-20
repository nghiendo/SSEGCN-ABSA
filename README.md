# SSEGCN-ABSA
Code and datasets of our paper "SSEGCN: Syntactic and Semantic Enhanced Graph Convolutional Network for Aspect-based Sentiment Analysis" accepted by NAACL 2022.



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.

## Preparation

1. Download and unzip GloVe vectors(`glove.840B.300d.zip`) from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it into  `SSEGCN/glove` directory.

2. Prepare dataset with:

   `python preprocess_data.py`

   To build the separate KG3 artefacts for Node2Vec-style graph features:

   `python dataset/preprocess_kg3.py --dataset all`

   This generates only:
   `train_kg3.json`: train split with node vocab, graph edges, and per-record KG3 annotations
   `test_kg3.json`: test split filtered by the train node vocabulary to avoid leakage

3. Prepare vocabulary with:

   `sh build_vocab.sh`

## Training

To train the SSEGCN model, run:

`sh run.sh`

## Credits

The code and datasets in this repository are based on [DualGCN_ABSA](https://github.com/CCChenhao997/DualGCN-ABSA) .

