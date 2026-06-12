# AGENTS.md

## Scope
- This repo's current primary model is `SG-MBSC-ABSA`.
- `train.py` is the real entrypoint.

## Model Names
- Use `sgmbsc` for the current non-BERT model.
- Use `sgmbsc_bert` for the BERT/ALBERT-backed variant.
- `ssegcn` and `ssegcn_bert` are aliases to the upgraded SG-MBSC models in this repo, not the original baselines.
- The original baselines are `ssegcn_original` and `ssegcn_bert_original`.

## Architecture
- SG-MBSC-ABSA keeps the original SSEGCN encoders and replaces the old single classifier head with the multi-branch head in `models/sg_mbsc_absa.py`.
- The SG head has:
  - one shared branch
  - three sentiment-specific expert branches: `pos`, `neu`, `neg`
  - gating between each expert branch and the shared branch
  - a residual baseline classifier path from the original aspect-pooled representation
  - auxiliary shared cross-entropy
  - prototype-based contrastive loss
- Non-BERT SG-MBSC uses `models/ssegcn.py` for the encoder and `models/sg_mbsc_absa.py` for the head.
- BERT SG-MBSC uses `models/ssegcn_bert.py` for the encoder and `models/sg_mbsc_absa.py` for the head.
- The BERT path projects token states to size `100` before the SG head; the SG BERT head is wired to that dimension.

## Data Flow
- Runtime training reads:
  - `dataset/Restaurants_corenlp/train_write.json`
  - `dataset/Restaurants_corenlp/test_write.json`
  - `dataset/Laptops_corenlp/train_write.json`
  - `dataset/Laptops_corenlp/test_write.json`
  - `dataset/Tweets_corenlp/train_write.json`
  - `dataset/Tweets_corenlp/test_write.json`
- Training does not read the raw `train.json` / `test.json` files directly.
- Non-BERT runs also expect vocab files inside the chosen dataset directory:
  - `vocab_tok.vocab`
  - `vocab_post.vocab`
  - `vocab_pos.vocab`
  - `vocab_dep.vocab`
  - `vocab_pol.vocab`

## Working Directory
- Run `train.py` from the repo root.
- `train.py` uses root-relative paths for:
  - `dataset/`
  - `glove/`
  - `log/`
  - `state_dict/`
- `dataset/preprocess_data.py` must be run from the `dataset/` directory because it opens `../dataset/...` paths.

## Commands
- Non-BERT laptop:
  - `python train.py --model_name sgmbsc --dataset laptop --vocab_dir ./dataset/Laptops_corenlp`
- Non-BERT restaurant:
  - `python train.py --model_name sgmbsc --dataset restaurant --vocab_dir ./dataset/Restaurants_corenlp`
- Non-BERT twitter:
  - `python train.py --model_name sgmbsc --dataset twitter --vocab_dir ./dataset/Tweets_corenlp`
- BERT/ALBERT example:
  - `python train.py --model_name sgmbsc_bert --dataset laptop --pretrained_bert_name albert-base-v2`
- Original baselines:
  - `python train.py --model_name ssegcn_original --dataset laptop --vocab_dir ./dataset/Laptops_corenlp`
  - `python train.py --model_name ssegcn_bert_original --dataset laptop --pretrained_bert_name albert-base-v2`

## Runtime Notes
- SG-MBSC training adds an auxiliary penalty only for models that expose `supports_contrastive = True`.
- That penalty is the sum of:
  - shared-branch CE loss
  - optional contrastive loss controlled by `--sg_cl_weight`
- Important label mapping inside the SG contrastive path:
  - dataset labels are `positive=0`, `negative=1`, `neutral=2`
  - branch order is `pos`, `neu`, `neg`
  - code remaps labels internally

## Verification
- There is no repo-local test, lint, CI, or typecheck config.
- The smallest meaningful verification is a focused train run with explicit `--model_name`, `--dataset`, and `--vocab_dir`.

## Existing Local Changes
- The worktree currently contains local edits in `README.md`, `ARCHITECTURE_ASCII.md`, `models/sg_mbsc_absa.py`, and `train.py`.
- Do not overwrite unrelated user changes.
