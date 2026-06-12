# SG-MBSC-ABSA Repository Architecture

This document describes the current repository with plain ASCII diagrams.

## 1. Repository Layout

```text
SSEGCN-ABSA/
|
|-- train.py
|   `-- Main CLI, dataset selection, SG-MBSC model selection, train/eval loop
|
|-- data_utils.py
|   `-- Dataset parsing, tokenization, embedding matrix, PyTorch Dataset classes
|
|-- prepare_vocab.py
|   `-- Vocabulary helper used by the non-BERT pipeline
|
|-- tree.py
|   `-- Dependency tree utilities
|
|-- layers.py
|   `-- Generic neural network layers kept from the base project
|
|-- models/
|   |-- ssegcn.py
|   |   `-- Original GloVe/LSTM/SSEGCN encoder and baseline classifier
|   |
|   |-- ssegcn_bert.py
|   |   `-- Original BERT/SSEGCN encoder and baseline classifier
|   |
|   `-- sg_mbsc_absa.py
|       `-- SG-MBSC-ABSA extension with multi-branch sentiment experts
|
|-- dataset/
|   |-- Restaurants_corenlp/
|   |   |-- train.json
|   |   |-- test.json
|   |   |-- train_write.json
|   |   `-- test_write.json
|   |
|   |-- Laptops_corenlp/
|   |   |-- train.json
|   |   |-- test.json
|   |   |-- train_write.json
|   |   `-- test_write.json
|   |
|   |-- Tweets_corenlp/
|   |   |-- train.json
|   |   |-- test.json
|   |   |-- train_write.json
|   |   `-- test_write.json
|   |
|   `-- preprocess_data.py
|
|-- glove/
|   `-- readme.md
|
|-- run.sh
|-- build_vocab.sh
|-- requirements.txt
|-- README.md
`-- CODE_ISSUES_REPORT.md
```

## 2. Top-Level Runtime Flow

```text
+------------------+
| Command line     |
| train.py args    |
+--------+---------+
         |
         v
+------------------+
| Select dataset   |
| restaurant       |
| laptop           |
| twitter          |
+--------+---------+
         |
         v
+------------------------------+
| Select model_name            |
| sgmbsc                       |
| sgmbsc_bert                  |
| ssegcn -> sgmbsc alias       |
| ssegcn_original baseline     |
+--------+---------------------+
         |
         v
+------------------------------+
| Build tokenizer and dataset  |
| - SentenceDataset            |
| - ABSAGCNData                |
+--------+---------------------+
         |
         v
+------------------------------+
| DataLoader                   |
| - train_dataloader           |
| - test_dataloader            |
+--------+---------------------+
         |
         v
+------------------------------+
| Instructor._train            |
| - forward                    |
| - CE loss                    |
| - optional contrastive loss  |
| - optimizer step             |
+--------+---------------------+
         |
         v
+------------------------------+
| Instructor._evaluate         |
| - accuracy                   |
| - macro F1                   |
| - report/confusion matrix    |
+------------------------------+
```

## 3. Dataset Pipeline

```text
+------------------------------+
| dataset/*_corenlp/*.json     |
| sentence, aspect, label, POS  |
| head, deprel, short distance |
+---------------+--------------+
                |
                v
+------------------------------+
| ParseData                    |
| - text tokens                |
| - aspect span                |
| - polarity string            |
| - aspect mask                |
| - position features          |
| - short path mask            |
+---------------+--------------+
                |
                +----------------------------------+
                |                                  |
                v                                  v
+------------------------------+   +------------------------------+
| Non-BERT path                |   | BERT path                    |
| SentenceDataset              |   | ABSAGCNData                  |
| - word ids                   |   | - BERT token ids             |
| - POS ids                    |   | - segment ids                |
| - position ids               |   | - attention mask             |
| - aspect mask                |   | - src/aspect mask            |
| - short_mask                 |   | - short_mask                 |
+------------------------------+   +------------------------------+
```

Label ids used by the datasets:

```text
positive -> 0
negative -> 1
neutral  -> 2
```

## 4. Original SSEGCN Encoder Flow

### 4.1 Non-BERT SSEGCN

```text
+------------------------------+
| Inputs                       |
| text, aspect, POS, position  |
| head, deprel, mask, length   |
| short_mask                   |
+---------------+--------------+
                |
                v
+------------------------------+
| Embedding                    |
| word + POS + position        |
+---------------+--------------+
                |
                v
+------------------------------+
| BiLSTM encoder               |
+---------------+--------------+
                |
                v
+------------------------------+
| Multi-head syntactic/semantic|
| attention adjacency          |
+---------------+--------------+
                |
                v
+------------------------------+
| GCN message passing          |
+---------------+--------------+
                |
                v
+------------------------------+
| Aspect mask pooling          |
+---------------+--------------+
                |
                v
+------------------------------+
| Baseline classifier          |
| available through            |
| ssegcn_original              |
+------------------------------+
```

### 4.2 BERT SSEGCN

```text
+------------------------------+
| BERT input ids               |
| segment ids, attention mask  |
| aspect span/masks, short mask|
+---------------+--------------+
                |
                v
+------------------------------+
| AutoModel                    |
| last_hidden_state            |
+---------------+--------------+
                |
                v
+------------------------------+
| Projection to attention dim  |
+---------------+--------------+
                |
                v
+------------------------------+
| Multi-head adjacency         |
+---------------+--------------+
                |
                v
+------------------------------+
| GCN message passing          |
+---------------+--------------+
                |
                v
+------------------------------+
| Aspect mask pooling          |
+---------------+--------------+
                |
                v
+------------------------------+
| Baseline classifier          |
| available through            |
| ssegcn_bert_original         |
+------------------------------+
```

## 5. SG-MBSC-ABSA Extension

The SG-MBSC models keep the existing encoders and replace the single pooled
classifier head with a shared-group multi-branch sentiment head. They also keep
the original aspect-pooling representation as a residual classifier path.

```text
+------------------------------+
| Encoder sequence features X  |
| shape: batch x seq x hidden  |
+---------------+--------------+
                |
                v
+----------------------------------------------------+
| Sentiment expert projections                       |
|                                                    |
| X -> Positive branch -> H_pos                      |
| X -> Neutral branch  -> H_neu                      |
| X -> Negative branch -> H_neg                      |
| X -> Shared branch   -> H_shared                   |
+---------------------------+------------------------+
                            |
                            v
+----------------------------------------------------+
| Local attention pooling                            |
|                                                    |
| H_pos    -> v_pos                                  |
| H_neu    -> v_neu                                  |
| H_neg    -> v_neg                                  |
| H_shared -> v_shared                               |
+---------------------------+------------------------+
                            |
                            v
+----------------------------------------------------+
| Cross-talk gating                                  |
|                                                    |
| Shared -> experts:                                 |
|   v_pos + v_shared -> gated positive vector        |
|   v_neu + v_shared -> gated neutral vector         |
|   v_neg + v_shared -> gated negative vector        |
|                                                    |
| Experts -> shared:                                 |
|   v_pos + v_neu + v_neg -> shared feedback         |
+---------------------------+------------------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
+---------------------------+   +---------------------------+
| Fusion classifier         |   | Contrastive objective     |
| concat all gated vectors  |   | shared vector as anchor   |
| -> logits                 |   | target branch from label  |
+-------------+-------------+   +-------------+-------------+
              |                           |
              v                           |
+---------------------------+             |
| Original SSEGCN residual  |             |
| aspect pooling -> logits  |             |
+-------------+-------------+             |
              |                           |
              +-------------+-------------+
                            |
                            v
+----------------------------------------------------+
| Training loss                                       |
| CE(fused logits, label) + sg_cl_weight * InfoNCE    |
+----------------------------------------------------+
```

### 5.1 SG-MBSC Model Names

```text
sgmbsc
  encoder: models.ssegcn.GCN
  head:    SGMBSCHead

sgmbscbert
  encoder: models.ssegcn_bert.GCNBert
  head:    SGMBSCHead

ssegcn
  alias:   sgmbsc

ssegcn_bert
  alias:   sgmbsc_bert

ssegcn_original
  encoder/head: original models.ssegcn.SSEGCNClassifier

ssegcn_bert_original
  encoder/head: original models.ssegcn_bert.SSEGCNBertClassifier
```

### 5.2 Contrastive Branch Mapping

The dataset label order differs from the expert branch order.

```text
Dataset label order:
  positive -> 0
  negative -> 1
  neutral  -> 2

SG-MBSC branch order:
  positive -> 0
  neutral  -> 1
  negative -> 2

Remap used for contrastive target:
  label 0 -> branch 0
  label 1 -> branch 2
  label 2 -> branch 1
```

## 6. Training Commands

```text
python train.py --model_name sgmbsc --dataset restaurant
python train.py --model_name sgmbsc --dataset laptop
python train.py --model_name sgmbsc --dataset twitter

python train.py --model_name sgmbsc_bert --dataset restaurant --pretrained_bert_name albert-base-v2

python train.py --model_name ssegcn_original --dataset laptop
python train.py --model_name ssegcn_bert_original --dataset laptop --pretrained_bert_name albert-base-v2
```

## 7. SG-MBSC Tunable Arguments

```text
--sg_expert_dim     Branch projection size. 0 keeps encoder hidden size.
--sg_temperature    Temperature for InfoNCE similarity logits.
--sg_cl_weight      Weight of the contrastive loss term.
--sg_dropout        Dropout before the final fusion classifier.
--sg_base_weight    Weight of the original SSEGCN residual logits.
```

## 8. Code Ownership Map

```text
train.py
  owns:
    - CLI arguments
    - model registry
    - dataset registry
    - optimizer selection
    - train/evaluate/test loop

data_utils.py
  owns:
    - reading JSON datasets
    - converting labels to ids
    - building token ids and masks
    - building BERT inputs

models/ssegcn.py
  owns:
    - non-BERT SSEGCN encoder
    - original non-BERT classifier

models/ssegcn_bert.py
  owns:
    - BERT-backed SSEGCN encoder
    - original BERT classifier

models/sg_mbsc_absa.py
  owns:
    - SG-MBSC expert branches
    - cross-talk gating
    - shared feedback
    - InfoNCE contrastive loss
    - SG-MBSC classifier wrappers
```
