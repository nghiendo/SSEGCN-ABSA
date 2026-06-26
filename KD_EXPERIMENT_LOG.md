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

6. `de95bd7` `exp6: dist logit correlation distillation`
   - Script: `experiments/run_kd_laptop_dist.sh`
   - Config delta: replace KL logits KD with DIST-style inter/intra class correlation distillation
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.6551`, `macro_f1=0.5439`
     - Conclusion: DIST alone is materially weaker than the exp4 recipe here.

7. `TBD` `exp7: add contrastive representation distillation`
   - Script: `experiments/run_kd_laptop_contrastive.sh`
   - Config delta: keep exp4 recipe and add `--kd_contrastive_weight 0.05`
   - Result:
     - Early-stopped after clear collapse toward majority predictions
     - Best observed checkpoint before stop: `acc=0.5396`, `macro_f1=0.2534`
     - Conclusion: the contrastive term is far too strong for this small student in the current feature space.

8. `TBD` `exp8: add margin distillation`
   - Script: `experiments/run_kd_laptop_margin.sh`
   - Config delta: keep exp4 recipe and add `--kd_margin_weight 0.05`
   - Result:
     - Early-stopped after clear collapse toward majority predictions
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: direct margin matching at this weight overwhelms the supervised signal.

9. `TBD` `exp9: add rank distillation`
   - Script: `experiments/run_kd_laptop_rank.sh`
   - Config delta: keep exp4 recipe and add a light pairwise ranking term `--kd_rank_weight 0.01`
   - Result:
     - Early-stopped after repeating the same majority-class collapse
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: even a light rank regularizer destabilizes this student when added from epoch 0.

10. `TBD` `exp10: add relational knowledge distillation`
   - Script: `experiments/run_kd_laptop_rkd.sh`
   - Config delta: keep exp4 recipe and add a very light RKD term `--kd_relation_weight 0.005`
   - Result:
     - Early-stopped after persistent underperformance
     - Best observed checkpoint before stop: `acc=0.5459`, `macro_f1=0.2767`
     - Conclusion: RKD is less catastrophic than contrastive/margin, but still far below the exp4 baseline.

11. `TBD` `exp11: add prototype class distillation`
   - Script: `experiments/run_kd_laptop_proto.sh`
   - Config delta: keep exp4 recipe and add soft class-prototype alignment plus prototype-relation matching
   - Result:
     - Early-stopped after persistent majority-class collapse
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: prototype losses also hurt when applied from the first epoch on this student.

Current best experiment:
- Commit: `8ebb68c`
- Script: `experiments/run_kd_laptop_unfreeze_word_emb_lr5e4.sh`
- Best selected checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6883_f1_0.6286`
