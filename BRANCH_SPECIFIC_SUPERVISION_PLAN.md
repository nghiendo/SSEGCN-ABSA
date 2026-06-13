# Branch-Specific Supervision Plan

## Goal

- Make the three sentiment experts `pos`, `neu`, and `neg` more specialized.
- Improve class separation, especially for the difficult `neutral` class.
- Keep the change parameter-efficient and local to the SG-MBSC head.

## Scope

- Only modify `models/sg_mbsc_absa.py`.
- Do not change:
  - `train.py`
  - `data_utils.py`
  - `models/sg_mbsc_encoder.py`
  - `models/sg_mbsc_bert_encoder.py`
  - dataset or preprocessing pipeline

## Current Head Structure

The current `SGMBSCHead` already has:

- one `shared` branch
- three sentiment-specific expert branches:
  - `pos`
  - `neu`
  - `neg`
- gating between each expert and the shared branch
- a shared-branch auxiliary CE loss
- a prototype-based contrastive loss

This means branch-specific supervision fits the current architecture rather than introducing a new concept.

## Main Idea

Add a lightweight auxiliary supervision signal so that:

- the correct expert branch responds more strongly for its own class
- the other two expert branches respond less strongly

This supervision should be applied to the private expert representations before gating, not after shared-expert mixing.

The preferred supervision targets are:

- `pooled["pos"]`
- `pooled["neu"]`
- `pooled["neg"]`

These are better supervision points because they are already pooled branch-level representations, but they have not yet been mixed with the shared branch through gating.

## Design

### 1. Add lightweight branch heads

Add one small scalar head for each expert:

- `pos_score`
- `neu_score`
- `neg_score`

Each head maps its private branch representation to a single logit.

### 2. Build branch targets

Supervise branch identity as a single-label 3-way auxiliary classification task.

### 3. Important label mapping

Dataset labels are:

- `positive = 0`
- `negative = 1`
- `neutral = 2`

Expert branch order is:

- `pos = 0`
- `neu = 1`
- `neg = 2`

So the correct remap is:

```text
[0, 2, 1]
```

Meaning:

- positive -> `pos`
- negative -> `neg`
- neutral -> `neu`

This remap must match the existing contrastive path.

### 4. Branch loss

Use `CrossEntropyLoss` on the three branch scores.

This is the cleaner default because the auxiliary task is naturally a 1-of-3 branch selection problem, not a multi-label problem.

The three scalar branch scores are concatenated into one 3-dimensional vector:

```text
[s_pos, s_neu, s_neg]
```

and the target is the remapped branch index.

`BCEWithLogitsLoss` remains a possible fallback if a softer one-vs-rest regularization is desired later, but it should not be the default first choice.

## Total Loss

The total training objective becomes:

```text
L_total = L_main + L_shared + L_contrastive + lambda_branch * L_branch
```

Where:

- `L_main`: final prediction cross-entropy
- `L_shared`: current shared-branch CE
- `L_contrastive`: current prototype contrastive loss
- `L_branch`: new branch-specific supervision loss on private expert branches

## Suggested Hyperparameter

Add a new coefficient:

- `lambda_branch`

Initial values to try:

- `0.02`
- `0.05`
- `0.1`

Recommended starting point:

- `0.05`

## Why This Is Feasible

### Architectural consistency

- The model already treats `pos`, `neu`, and `neg` as sentiment-specific experts.
- The current contrastive loss already assumes branch-label alignment.

### Minimal implementation cost

- Only the SG head needs to change.
- The trainer already supports an auxiliary penalty returned by the model.

### Low parameter overhead

- Only a few tiny linear heads are added.

## Risks

### 1. Wrong label-branch mapping

If the remap is wrong, the model will silently supervise the wrong expert.

### 2. Over-constraining the experts

If `lambda_branch` is too large:

- experts may become too rigid
- shared-expert cooperation may weaken
- generalization may drop

### 3. Interaction with gating

If supervision is applied after gating, the model may partially rely on shared information instead of forcing the private experts to become more specialized.

This is why the plan uses pre-gating private branch representations for supervision.

### 4. Auxiliary loss imbalance

The model already has:

- final CE
- shared CE
- contrastive loss

The new branch loss should remain a small auxiliary term.

## Risk Mitigation

- keep `lambda_branch` small at first
- keep the final inference path unchanged
- reuse the same label remap already used by the contrastive loss
- supervise private branch representations before gating
- run clean ablations before combining with more changes

## Experimental Plan

### B0. Baseline

- current `sgmbsc` or `sgmbsc_bert`

### B1. Baseline + branch-specific supervision

- add branch heads
- add `L_branch`

### B2. Tune branch loss weight

Try:

- `lambda_branch = 0.02`
- `lambda_branch = 0.05`
- `lambda_branch = 0.1`

## Evaluation Metrics

- accuracy
- macro F1
- confusion matrix
- per-class F1
- especially `neutral` F1

## Success Criteria

- macro F1 improves consistently
- neutral performance improves or at least does not regress
- parameter count increase is negligible
- training remains stable

## Keywords

- branch-specific supervision
- class-specific expert supervision
- sentiment-specific experts
- expert specialization
- auxiliary branch loss
- branch identity supervision
- class-aware expert learning
- one-vs-rest branch supervision
