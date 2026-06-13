# SG-MBSC-ABSA Repository Architecture

This document describes the current repository after removing the legacy model
family.

## 1. Runtime Flow

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
| - auxiliary SG losses        |
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

## 2. Dataset Pipeline

```text
+------------------------------+
| dataset/*_corenlp/*.json     |
| sentence, aspect, label, POS |
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

## 3. Encoder Paths

### 3.1 Non-BERT `sgmbsc`

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
| Multi-head adjacency         |
| aspect-aware + short_mask    |
+---------------+--------------+
                |
                v
+------------------------------+
| SGMBSCEncoder                |
| models/sg_mbsc_encoder.py    |
+------------------------------+
```

### 3.2 BERT `sgmbsc_bert`

```text
+------------------------------+
| BERT input ids               |
| segment ids, attention mask  |
| aspect span/masks, short mask|
+---------------+--------------+
                |
                v
+------------------------------+
| AutoModel + LayerNorm        |
+---------------+--------------+
                |
                v
+------------------------------+
| Projection to size 100       |
+---------------+--------------+
                |
                v
+------------------------------+
| SGMBSCBertEncoder            |
| models/sg_mbsc_bert_encoder.py|
+------------------------------+
```

## 4. SG Head

```text
+------------------------------+
| Encoder sequence features X  |
| shape: batch x seq x hidden  |
+---------------+--------------+
                |
                v
+----------------------------------------------------+
| Sentiment expert projections                       |
| X -> pos, neu, neg, shared                         |
+---------------------------+------------------------+
                            |
                            v
+----------------------------------------------------+
| Masked local attention pooling                     |
| v_pos, v_neu, v_neg, v_shared                      |
+---------------------------+------------------------+
                            |
                            v
+----------------------------------------------------+
| Cross-talk gating                                  |
| gated expert vectors mixed with shared branch      |
+---------------------------+------------------------+
                            |
                            v
+----------------------------------------------------+
| Joint embedding                                    |
| concat(pos, neu, neg, shared) -> projected latent  |
+---------------------------+------------------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
+---------------------------+   +---------------------------+
| Final classifier          |   | Auxiliary losses          |
| joint + shared + residual |   | shared CE + contrastive  |
+-------------+-------------+   +-------------+-------------+
              |                           |
              +-------------+-------------+
                            |
                            v
+----------------------------------------------------+
| Total loss                                         |
| CE(final logits) + SG auxiliary penalties          |
+----------------------------------------------------+
```

Contrastive branch mapping:

```text
Dataset labels:
  positive -> 0
  negative -> 1
  neutral  -> 2

Branch order:
  positive -> 0
  neutral  -> 1
  negative -> 2

Remap:
  label 0 -> branch 0
  label 1 -> branch 2
  label 2 -> branch 1
```

## 5. Training Commands

```text
python train.py --model_name sgmbsc --dataset restaurant
python train.py --model_name sgmbsc --dataset laptop
python train.py --model_name sgmbsc --dataset twitter

python train.py --model_name sgmbsc_bert --dataset restaurant --pretrained_bert_name albert-base-v2
```

## 6. Key Files

```text
train.py
  - CLI arguments
  - model registry
  - train/evaluate/test loop

models/sg_mbsc_encoder.py
  - non-BERT SG-MBSC encoder

models/sg_mbsc_bert_encoder.py
  - BERT-backed SG-MBSC encoder

models/sg_mbsc_absa.py
  - multi-branch SG head
  - joint embedding fusion
  - auxiliary losses
  - classifier wrappers
```
