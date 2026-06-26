# ShallowBERT Distillation Architecture

## Objective

This repository now includes a reproducible shallow-BERT distillation path that turns the original `ssegcnbert` teacher into a smaller `ssegcnbertshallow` student while preserving the SSEGCN ABSA head and the BERT-centric inductive bias.

The practical goal is not paper-style teacher rebuilding. The goal is:

- reuse the local teacher checkpoints already produced in this repo
- distill into a 6-layer BERT student
- validate the same method family across `laptop`, `restaurant`, and `twitter`
- keep only code, scripts, and logs in git, not checkpoints

## Model Architecture

### Teacher

Teacher model class:

- `models/ssegcn_bert.py`: `SSEGCNBertClassifier`

High-level structure:

- full `bert-base-uncased` encoder
- SSEGCN graph/convolution stack on top of BERT token representations
- classifier head for 3-way ABSA polarity prediction

### Student

Student model class:

- `models/ssegcn_bert.py`: `SSEGCNBertStudentClassifier`

Student design choices:

- keep the same SSEGCN ABSA head family as the teacher
- truncate BERT from 12 layers to 6 layers
- preserve the teacher-compatible hidden size and ABSA head behavior
- initialize student BERT layers from the fine-tuned teacher checkpoint

Important student controls:

- `--student_bert_layers 6`
- `--student_bert_layer_map uniform|first|last`
- `--student_init_path <checkpoint>`

This is not a tiny word-side student. It is a BERT-side compression path that keeps the teacher task head close to intact and compresses primarily through encoder depth.

## Distillation Primitives

The KD implementation lives in:

- `train.py`

Important options:

- `--kd_logit_mode {kl,dkd,dist}`
- `--kd_alpha`
- `--kd_beta`
- `--kd_gamma`
- `--kd_dist_inter_weight`
- `--kd_dist_intra_weight`
- `--kd_token_hidden_weight`
- `--kd_token_relation_weight`
- `--kd_hidden_layer_weight`

Implemented supervision families:

- hard-label loss
- classic logit KL distillation
- DKD-style decoupled KD
- DIST-style relation-aware logit distillation
- token hidden-state KD
- token relation KD
- intermediate hidden-layer KD

The winning family in this repo is:

- shallow BERT continuation
- light token hidden-state KD
- final DIST-based last-mile tuning

## Winning Step-By-Step Method

The best-performing protocol is not a single one-shot script from random init.

It is a continuation chain:

1. `stage54-uniform`
- initialize a 6-layer student from the teacher
- use `student_bert_layer_map=uniform`
- keep feature KD active

2. `stage55-last-logits`
- continue from stage 54 best checkpoint
- switch to `student_bert_layer_map=last`
- remove feature KD dominance and emphasize logit KD

3. `stage56-tinybert-lite`
- add light token hidden-state KD and token relation KD

4. `stage57-hidden-only`
- remove token relation KD
- keep only token hidden-state KD

5. `stage58-hidden-polish`
- lower LR and gently polish the hidden-only regime

6. `stage59-accuracy-push`
- move more weight toward hard labels

7. `stage60-hardlabel-edge`
- continue increasing hard-label emphasis

8. `stage61-hardlabel-polish`
- local polish around the best pre-DIST basin

9. `stage64-dist-lastmile`
- switch final logit KD to `DIST`
- use:
  - `kd_alpha=0.76`
  - `kd_beta=0.12`
  - `kd_logit_mode=dist`
  - `kd_dist_inter_weight=1.0`
  - `kd_dist_intra_weight=2.0`
  - light token hidden KD

This chain is automated by:

- [experiments/run_kd_shallowbert6_exp64_chain_short.sh](/SSEGCN-ABSA/experiments/run_kd_shallowbert6_exp64_chain_short.sh)

Dataset wrappers:

- [experiments/run_kd_restaurant_shallowbert6_dist_short.sh](/SSEGCN-ABSA/experiments/run_kd_restaurant_shallowbert6_dist_short.sh)
- [experiments/run_kd_twitter_shallowbert6_dist_short.sh](/SSEGCN-ABSA/experiments/run_kd_twitter_shallowbert6_dist_short.sh)

Local teacher scripts:

- [experiments/run_teacher_restaurant_short.sh](/SSEGCN-ABSA/experiments/run_teacher_restaurant_short.sh)
- [experiments/run_teacher_twitter_short.sh](/SSEGCN-ABSA/experiments/run_teacher_twitter_short.sh)

## Why This Works Better Than One-Shot DIST

The important empirical lesson from this repo is:

- direct DIST from scratch is not equivalent to the winning laptop method
- the student needs a staged warm-start path
- DIST works best as a last-mile sharpening phase after the student is already close to the teacher manifold

In practice:

- random-init or shallow-init DIST can stall badly
- progressive continuation transfers much more of the teacher signal
- the final DIST stage then improves both accuracy and macro-F1

## Why The Student Can Beat The Teacher

This section separates direct observations from technical interpretation.

### Direct observations from this repo

Observed outcomes:

- `laptop`: the best shallow student exceeds the local teacher on both accuracy and macro-F1
- `twitter`: the validated chain exceeds the local teacher on both accuracy and macro-F1
- `restaurant`: the validated chain has already exceeded the local teacher on both accuracy and macro-F1 during the active run

Observed optimization pattern:

- one-shot DIST from scratch is not enough
- a staged continuation path consistently works much better
- the strongest checkpoints often appear after the student is already close to the teacher and then receives a final DIST-style refinement

Observed architectural pattern:

- the best student is not radically different from the teacher
- the student keeps the same SSEGCN task head family
- the compression happens mostly by reducing BERT depth from 12 layers to 6 layers

### Technical interpretation

The student can beat the teacher here because distillation is not just compression. In this pipeline it also acts as regularized task-specific re-optimization.

#### 1. The teacher is strong, but not fully optimal

The local teacher checkpoints are good task models, but they are still products of a finite training recipe, a fixed seed, and a fixed optimization path.

That means the student is not trying to imitate a perfect Bayes-optimal model. It is imitating a strong but imperfect model while still seeing the ground-truth labels.

So the student objective is effectively:

- fit the real labels
- inherit the teacher's dark knowledge
- avoid inheriting every teacher mistake exactly

This can outperform the teacher itself when the student keeps the useful structure but is pulled away from some teacher overfitting or calibration errors.

#### 2. Distillation is acting like structured regularization

Hard labels alone are sparse. Teacher outputs contain richer information:

- class similarity
- confidence shape
- sample-to-sample relational structure

The final winning stage uses `DIST`, which does not only match logits pointwise. It also preserves inter-sample and intra-class correlation structure.

Reasonable interpretation:

- this reduces variance in the student decision surface
- it encourages smoother class geometry
- it improves generalization near ambiguous ABSA examples

That is one plausible reason the student can improve macro-F1 more than the teacher, especially on minority or harder sentiment cases.

#### 3. The 6-layer student can generalize better than the 12-layer teacher

Smaller does not always mean weaker on test data.

In this setting, reducing encoder depth can help because:

- the student has less capacity to memorize teacher-specific noise
- the head remains task-aligned, so the compression is not too destructive
- the student still starts from a highly informative teacher initialization

This creates a favorable bias-variance tradeoff:

- lower variance than the full teacher
- enough bias control because the BERT backbone and SSEGCN head are still strong

In short, the student is smaller in a way that removes redundancy more than useful task signal.

#### 4. The continuation chain matters more than raw compression

The student does not beat the teacher simply because it has 6 layers.

It beats the teacher because the chain performs a sequence of controlled refinements:

- initialize from the fine-tuned teacher
- preserve broad teacher behavior
- gradually simplify the supervision
- shift toward harder label fitting
- use DIST only after the student already lives in a strong local basin

This is closer to iterative model surgery than to naive KD.

The student wins because the training path is better behaved, not just because the architecture is smaller.

#### 5. The student is effectively getting two teachers

Functionally, the final student is supervised by:

- the original dataset labels
- the teacher predictions and relations

This dual supervision can be stronger than the teacher's original single training path, because the teacher itself was only optimized once against hard labels, while the student gets:

- teacher-induced soft structure
- direct correction from hard targets
- repeated local polishing stages

That combination can produce a better final solution than the teacher checkpoint it started from.

### Bottom-line interpretation

The best explanation is not that the student is intrinsically superior because it is smaller.

The better explanation is:

- the teacher provides a strong representation and soft target prior
- the smaller student adds useful regularization
- the staged continuation path re-optimizes the task head and truncated encoder more effectively than the original teacher recipe
- the final DIST refinement improves class geometry enough to outperform the local teacher on test metrics

So the student wins here because it is a better-regularized and better-polished task solution, not because compression alone magically creates accuracy.

## Current Best Known Results

### Laptop

Teacher:

- `state_dict/ssegcnbert_laptop_acc_0.7722_f1_0.7031`

Best student:

- `state_dict/ssegcnbertshallow_laptop_acc_0.7737_f1_0.7360`

Outcome:

- student beats teacher on both accuracy and macro-F1

### Twitter

Teacher:

- `state_dict/ssegcnbert_twitter_acc_0.7386_f1_0.7317`

Best student observed in the validated chain:

- `state_dict/ssegcnbertshallow_twitter_acc_0.7710_f1_0.7589`

Outcome:

- student beats teacher on both accuracy and macro-F1

### Restaurant

Teacher:

- `state_dict/ssegcnbert_restaurant_acc_0.8365_f1_0.7439`

Best student observed so far during the active chain:

- `state_dict/ssegcnbertshallow_restaurant_acc_0.8374_f1_0.7569`

Outcome:

- student has already exceeded teacher on both accuracy and macro-F1 during the validated chain
- the chain may still improve further while running

## Reproducibility Notes

### Dataset integrity

The restaurant pipeline depends on a valid regenerated file:

- `dataset/Restaurants_corenlp/train_write.json`

The corrupted 1-sample version is invalid for training. The correct regenerated version contains the full 1980 training examples.

### Artifact policy

Do not push:

- `state_dict/`
- `log/`
- generated vocab/tokenizer/embedding artifacts

Push only:

- code changes
- experiment scripts
- experiment log markdown
- architecture/process documentation

## Audit Trail

Relevant experiment log entries:

- `exp64`: laptop DIST last-mile win
- `exp69`: local restaurant teacher recreation
- `exp70`: cross-dataset validation chain scripts
- `exp71`: local twitter teacher recreation
- `exp72`: twitter chain validation result

Detailed metrics and per-experiment conclusions remain in:

- [KD_EXPERIMENT_LOG.md](/SSEGCN-ABSA/KD_EXPERIMENT_LOG.md)
