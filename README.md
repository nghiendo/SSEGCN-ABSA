# SG-MBSC-ABSA
This repository now uses SG-MBSC-ABSA as the default model. SG-MBSC-ABSA keeps
the original SSEGCN encoder and adds shared-group multi-branch sentiment
experts, cross-talk gating, a contrastive objective, and a residual baseline
classifier path from the original aspect-pooling representation.



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

To train the current SG-MBSC-ABSA model, run:

`sh run.sh`

For a plain ASCII architecture map of this repository, see
`ARCHITECTURE_ASCII.md`.

The original model name `ssegcn` is kept as an upgraded alias for `sgmbsc`, so
older commands now train the new model. For explicit commands, use:

```powershell
python train.py --model_name sgmbsc --dataset restaurant
python train.py --model_name sgmbsc --dataset laptop
python train.py --model_name sgmbsc --dataset twitter
```

For the BERT-backed encoder path:

```powershell
python train.py --model_name sgmbsc_bert --dataset restaurant --pretrained_bert_name albert-base-v2
```

The SG-MBSC head exposes `--sg_expert_dim`, `--sg_temperature`,
`--sg_cl_weight`, `--sg_dropout`, and `--sg_base_weight` for the branch
dimension, InfoNCE temperature, contrastive-loss weight, classifier dropout, and
the residual weight of the original SSEGCN aspect-pooling classifier.

To run the untouched original SSEGCN baseline for comparison:

```powershell
python train.py --model_name ssegcn_original --dataset laptop --vocab_dir ./dataset/Laptops_corenlp
python train.py --model_name ssegcn_bert_original --dataset laptop --pretrained_bert_name albert-base-v2
```

## Credits

The code and datasets in this repository are based on [DualGCN_ABSA](https://github.com/CCChenhao997/DualGCN-ABSA) .

