# KD Experiment Log

Metric priority used for selecting the strongest experiment:
- Primary: macro F1 on the test set
- Secondary: test accuracy

Teacher setup used for all KD runs:
- Dataset: `laptop`
- Teacher checkpoint: `state_dict/ssegcnbert_laptop_acc_0.7722_f1_0.7031`
- Teacher run summary: `max_test_acc=0.7722`, `max_f1=0.7161`

Non-experiment prep commit:
- `57bc686` `prep: allow random student embeddings without glove`
  - Purpose: enable student KD runs in this environment without downloading GloVe.

Experiments:

1. `94dd44e` `exp1: baseline kd student on laptop`
   - Script: `experiments/run_kd_laptop_baseline.sh`
   - Config delta: baseline student KD with frozen word embeddings
   - Result:
     - Selected checkpoint: `acc=0.6614`, `macro_f1=0.5814`
     - Training observed `max_f1=0.5930`, but it was not retained because checkpoint selection was accuracy-based.

2. `b5d2ddd` `exp2: select kd checkpoint by macro f1`
   - Script: `experiments/run_kd_laptop_baseline.sh`
   - Config delta: save KD checkpoints by macro F1, break ties with accuracy
   - Result:
     - Selected checkpoint: `acc=0.6614`, `macro_f1=0.5930`
     - Improvement over exp1: same accuracy, better retained F1

3. `02dcffe` `exp3: unfreeze random student word embeddings`
   - Script: `experiments/run_kd_laptop_unfreeze_word_emb.sh`
   - Config delta: `--student_freeze_word_emb false`
   - Result:
     - Selected checkpoint: `acc=0.6851`, `macro_f1=0.6237`
     - Training observed `max_test_acc=0.6883`

4. `8ebb68c` `exp4: lower lr for unfrozen student embeddings`
   - Script: `experiments/run_kd_laptop_unfreeze_word_emb_lr5e4.sh`
   - Config delta: `--student_lr 5e-4` on top of exp3
   - Result:
     - Selected checkpoint: `acc=0.6883`, `macro_f1=0.6286`
     - Best experiment so far

5. `3a06886` `exp5: hidden-feature dkd with logit standardization`
   - Script: `experiments/run_kd_laptop_hidden_dkd.sh`
   - Config delta: switch to `--kd_logit_mode dkd`, `--kd_logit_standardize true`, and `--kd_feature_mode teacher_hidden`
   - Result:
     - Selected checkpoint: `acc=0.6614`, `macro_f1=0.5702`
     - Conclusion: hidden-feature matching to the teacher encoder hurt this student.

6. `TBD` `exp6: dist logit correlation distillation`
   - Script: `experiments/run_kd_laptop_dist.sh`
   - Config delta: replace KL logits KD with DIST-style inter/intra class correlation distillation
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.6551`, `macro_f1=0.5439`
     - Conclusion: DIST alone is materially weaker than the exp4 recipe here.

Current best experiment:
- Commit: `8ebb68c`
- Script: `experiments/run_kd_laptop_unfreeze_word_emb_lr5e4.sh`
- Best selected checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6883_f1_0.6286`
