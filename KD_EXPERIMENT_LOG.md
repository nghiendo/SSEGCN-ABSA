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
- `e3be955` `prep: persist best checkpoints during training`
  - Purpose: save each best-so-far checkpoint immediately so interrupted KD runs still leave a usable artifact in `state_dict/`.

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

7. `f51e0ea` `exp7: add contrastive representation distillation`
   - Script: `experiments/run_kd_laptop_contrastive.sh`
   - Config delta: keep exp4 recipe and add `--kd_contrastive_weight 0.05`
   - Result:
     - Early-stopped after clear collapse toward majority predictions
     - Best observed checkpoint before stop: `acc=0.5396`, `macro_f1=0.2534`
     - Conclusion: the contrastive term is far too strong for this small student in the current feature space.

8. `f51e0ea` `exp8: add margin distillation`
   - Script: `experiments/run_kd_laptop_margin.sh`
   - Config delta: keep exp4 recipe and add `--kd_margin_weight 0.05`
   - Result:
     - Early-stopped after clear collapse toward majority predictions
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: direct margin matching at this weight overwhelms the supervised signal.

9. `f51e0ea` `exp9: add rank distillation`
   - Script: `experiments/run_kd_laptop_rank.sh`
   - Config delta: keep exp4 recipe and add a light pairwise ranking term `--kd_rank_weight 0.01`
   - Result:
     - Early-stopped after repeating the same majority-class collapse
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: even a light rank regularizer destabilizes this student when added from epoch 0.

10. `f51e0ea` `exp10: add relational knowledge distillation`
   - Script: `experiments/run_kd_laptop_rkd.sh`
   - Config delta: keep exp4 recipe and add a very light RKD term `--kd_relation_weight 0.005`
   - Result:
     - Early-stopped after persistent underperformance
     - Best observed checkpoint before stop: `acc=0.5459`, `macro_f1=0.2767`
     - Conclusion: RKD is less catastrophic than contrastive/margin, but still far below the exp4 baseline.

11. `f51e0ea` `exp11: add prototype class distillation`
   - Script: `experiments/run_kd_laptop_proto.sh`
   - Config delta: keep exp4 recipe and add soft class-prototype alignment plus prototype-relation matching
   - Result:
     - Early-stopped after persistent majority-class collapse
     - Best observed checkpoint before stop: `acc=0.5332`, `macro_f1=0.2319`
     - Conclusion: prototype losses also hurt when applied from the first epoch on this student.

12. `9b3572b` `exp12: add distilbert-style kd curriculum`
   - Script: `experiments/run_kd_laptop_anneal_warmup.sh`
   - Config delta: CE warmup from scratch, then linear KD ramp with temperature annealing
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.6060`, `macro_f1=0.4397`
     - Conclusion: curriculum alone does not recover the ground lost by training this student from random init.

13. `3d78ed2` `exp13: add tinybert-style stage2 distillation`
   - Script: `experiments/run_kd_laptop_tinybert_stage2.sh`
   - Config delta: warm-start from the best student checkpoint and run a short second-stage KD pass with lower LR and cosine feature alignment
   - Result:
     - Selected checkpoint: `acc=0.6883`, `macro_f1=0.6325`
     - New best experiment so far

14. `f776101` + `250ffc1` `exp14: add minilm-style word relation distillation`
   - Script: `experiments/run_kd_laptop_minilm_stage2.sh`
   - Config delta: warm-start from the current best student and add word-level token relation distillation by aggregating teacher subwords back to original words
   - Result:
     - Early-stopped after plateauing below the current best
     - Best observed checkpoint before stop: `acc=0.6820`, `macro_f1=0.6231`
     - Conclusion: relation KD is stable after warm-starting, but still weaker than the TinyBERT-style stage-2 pass.

15. `c29b66c` `exp15: add mobilebert-style deeper bottleneck student`
   - Script: `experiments/run_kd_laptop_mobilebert_style.sh`
   - Config delta: use a thinner but deeper student with input bottleneck and two recurrent layers
   - Result:
     - Early-stopped after collapse from random initialization
     - Best observed checkpoint before stop: `acc=0.3861`, `macro_f1=0.2333`
     - Conclusion: the architecture change alone does not offset the optimization difficulty of a deeper student in this setup.

16. `1ddc04f` `exp16: add student teacher and assistant kd scripts`
   - Script: `experiments/run_kd_laptop_ban_stage3.sh`
   - Config delta: born-again/self-distillation stage using the best student as teacher and warm-starting from the strongest pre-stage-2 checkpoint
   - Result:
     - Early-stopped after plateauing below the current best
     - Best observed checkpoint before stop: `acc=0.6899`, `macro_f1=0.6314`
     - Conclusion: self-distillation was stable and slightly improved accuracy, but still missed the best F1 by a small margin.

17. `1ddc04f` `exp17: run assistant-wide student`
   - Script: `experiments/run_kd_laptop_assistant_wide.sh`
   - Config delta: train a wider intermediate student (`hidden_dim=48`, `pos/post_dim=12`) directly from the BERT teacher
   - Result:
     - Early-stopped after the wider student converged clearly below the current best
     - Best observed checkpoint before stop: `acc=0.6772`, `macro_f1=0.6122`
     - Conclusion: widening the student from random initialization improved over weak KD baselines, but not enough to justify a teacher-assistant chain from this checkpoint.

18. `7328761` `exp18: add patient stage2 token hidden distillation`
   - Script: `experiments/run_kd_laptop_patient_stage2.sh`
   - Config delta: warm-start from the best checkpoint and add a token-level hidden-state alignment loss against teacher word states, analogous to a Patient KD / TinyBERT hidden-state stage
   - Result:
     - Early-stopped after the new token projection destabilized the already-strong checkpoint
     - Best observed checkpoint before stop: `acc=0.6804`, `macro_f1=0.6218`
     - Conclusion: direct token hidden matching with a freshly initialized projection was too aggressive for late-stage fine-tuning.

19. `3b5b711` `exp19: add lite patient stage2 schedule`
   - Script: `experiments/run_kd_laptop_patient_stage2_lite.sh`
   - Config delta: keep the same token hidden loss but add a CE warmup, cosine KD ramp, lower LR, and much lighter token-hidden weight
   - Result:
     - Early-stopped after the best metric still came from the warmup phase
     - Best observed checkpoint before stop: `acc=0.6867`, `macro_f1=0.6303`
     - Conclusion: the gentler schedule preserved the checkpoint better, but token hidden KD still failed to beat the existing stage-2 baseline.

20. `2315da7` `exp20: add dual teacher stage2 distillation`
   - Script: `experiments/run_kd_laptop_dual_teacher_stage2.sh`
   - Config delta: warm-start from the best student, keep feature supervision from the BERT teacher, and blend logits from BERT plus the best student teacher
   - Result:
     - Early-stopped after the dual-teacher blend plateaued below the current best
     - Best observed checkpoint before stop: `acc=0.6835`, `macro_f1=0.6280`
     - Conclusion: a fixed logits blend was stable but did not create extra headroom beyond the strongest single-teacher stage-2 recipe.

21. `8d2be53` `exp21: add confidence-gated dual teacher distillation`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_stage2.sh`
   - Config delta: replace the fixed dual-teacher blend with a confidence-gated probability-space blend between the BERT teacher and the best student teacher
   - Result:
     - Early-stopped after confirming a small F1 improvement over the previous best
     - Best observed checkpoint before stop: `acc=0.6820`, `macro_f1=0.6339`
     - Caveat: this run happened before immediate checkpoint persistence, so the improved checkpoint was first verified from the training log and then reproduced in exp22
     - Conclusion: adaptive dual-teacher KD finally nudged macro F1 above the TinyBERT-style stage-2 baseline.

22. `d50d299` `exp22: add short confidence-gated dual teacher rerun`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_stage2_short.sh`
   - Config delta: rerun the confidence-gated recipe for a single epoch so the early peak is retained, with immediate checkpoint persistence enabled
   - Result:
     - Selected checkpoint: `acc=0.6820`, `macro_f1=0.6339`
     - New best experiment so far
     - Checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6820_f1_0.6339`

23. `2bef898` `exp23: add disagreement-gated dual teacher rerun`
   - Script: `experiments/run_kd_laptop_dual_teacher_disagreement_stage2_short.sh`
   - Config delta: continue from the new best checkpoint and switch the adaptive blend to disagreement-gated mixing
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6804`, `macro_f1=0.6293`
     - Conclusion: disagreement gating is weaker than confidence gating on top of the current best student.

24. `e367610` `exp24: add confidence-gated stage3 dual teacher rerun`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_stage3_short.sh`
   - Config delta: continue directly from the new best checkpoint with a smaller LR and slightly more KD-heavy weighting, while keeping the confidence-gated dual-teacher recipe
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6788`, `macro_f1=0.6306`
     - Conclusion: a conservative stage-3 polish pass did not improve on the confidence-gated stage-2 checkpoint.

25. `e437850` `exp25: add uncertainty-gated dual teacher rerun`
   - Script: `experiments/run_kd_laptop_dual_teacher_uncertainty_stage2_short.sh`
   - Config delta: continue from the current best checkpoint and gate the aux-teacher contribution by the BERT teacher's predictive uncertainty
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6772`, `macro_f1=0.6296`
     - Conclusion: uncertainty gating is weaker than the earlier confidence-gated blend on this dataset.

26. `b694f84` `exp26: add short stage4 born-again distillation`
   - Script: `experiments/run_kd_laptop_ban_stage4_short.sh`
   - Config delta: self-distill directly from the current best student teacher with a low-LR one-epoch born-again pass
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6772`, `macro_f1=0.6254`
     - Conclusion: stage-4 self-distillation underperforms here, largely because the student-teacher feature projection path is too disruptive even with light weighting.

Current best experiment:
- Commit: `d50d299`
- Script: `experiments/run_kd_laptop_dual_teacher_confidence_stage2_short.sh`
- Best selected checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6820_f1_0.6339`
