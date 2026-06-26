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

3. Prepare vocabulary with:

   `sh build_vocab.sh`

## Training

To train the SSEGCN model, run:

`sh run.sh`

## Distillation

The repository also contains a validated shallow-BERT distillation path that compresses the original BERT teacher into a 6-layer BERT student while keeping the SSEGCN ABSA head.

Main references:

- [docs/SHALLOWBERT_DISTILLATION_ARCHITECTURE.md](/SSEGCN-ABSA/docs/SHALLOWBERT_DISTILLATION_ARCHITECTURE.md)
- [KD_EXPERIMENT_LOG.md](/SSEGCN-ABSA/KD_EXPERIMENT_LOG.md)

Cross-dataset reproduction scripts:

- `bash experiments/run_kd_shallowbert6_exp64_chain_short.sh laptop`
- `bash experiments/run_kd_shallowbert6_exp64_chain_short.sh restaurant`
- `bash experiments/run_kd_shallowbert6_exp64_chain_short.sh twitter`

## Credits

The code and datasets in this repository are based on [DualGCN_ABSA](https://github.com/CCChenhao997/DualGCN-ABSA) .
