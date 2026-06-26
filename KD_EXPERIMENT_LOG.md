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

27. `ab08e7d` `exp27: add short dkd stage2 distillation`
   - Script: `experiments/run_kd_laptop_dkd_stage2_short.sh`
   - Config delta: keep the current best checkpoint as initialization, switch logit KD to DKD, and reduce non-logit feature pressure for a one-epoch warm-started stage-2 pass
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6317`
     - Conclusion: warm-started DKD is much more stable than the earlier from-scratch DKD run, but still does not beat the confidence-gated dual-teacher checkpoint on macro F1.

28. `454faa1` `exp28: add logits-only confidence dual teacher kd`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_logits_only_short.sh`
   - Config delta: continue from the current best checkpoint, keep the confidence-gated dual-teacher logits blend, and remove explicit feature KD pressure by setting `kd_gamma=0.0`
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6804`, `macro_f1=0.6320`
     - Conclusion: logits-only refinement stays close to the best run, but the small feature-alignment term from the winning recipe still appears helpful.

29. `959ea10` `exp29: add minilm dual-teacher short distillation`
   - Script: `experiments/run_kd_laptop_minilm_dual_teacher_short.sh`
   - Config delta: keep the confidence-gated dual-teacher recipe, drop direct feature KD, and add a very light MiniLM-style token relation loss
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6335`
     - Conclusion: warm-started relation KD is now stable and nearly matches the best checkpoint, but it still misses the `0.6339` peak by a small margin.

30. `814a878` `exp30: add standardized dual-teacher short distillation`
   - Script: `experiments/run_kd_laptop_dual_teacher_standardized_short.sh`
   - Config delta: keep the confidence-gated dual-teacher setup, add logit standardization, and retain only a light feature KD term
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6239`
     - Conclusion: standardizing logits removes useful scale information in this late-stage recipe and hurts macro F1 materially.

31. `be2b08f` `exp31: add tinybert stage2 best short distillation`
   - Script: `experiments/run_kd_laptop_tinybert_stage2_best_short.sh`
   - Config delta: run a teacher-only TinyBERT/DistilBERT-style one-epoch stage-2 refinement from the current best checkpoint
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6804`, `macro_f1=0.6280`
     - Conclusion: removing the auxiliary student teacher drops performance well below the best dual-teacher setup.

32. `be5ddb1` `exp32: add tinybert stage2 best short no-weight distillation`
   - Script: `experiments/run_kd_laptop_tinybert_stage2_best_short_noweight.sh`
   - Config delta: repeat the teacher-only short stage-2 pass but disable per-instance KD weighting
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6788`, `macro_f1=0.6268`
     - Conclusion: turning off instance weighting does not rescue the teacher-only stage-2 refinement.

33. `f0dfdbb` `exp33: add primary-teacher weighting for dual kd`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_primary_weight_short.sh`
   - Config delta: keep the best confidence-gated dual-teacher recipe, but restore meaningful per-instance weights by computing them from the primary BERT teacher before auxiliary blending
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6788`, `macro_f1=0.6289`
     - Conclusion: reactivating primary-teacher weighting changes the optimization behavior as intended, but it still underperforms the simpler unweighted confidence-gated blend.

34. `aae47ed` `exp34: add confidence dual-teacher best replay`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_best_replay_short.sh`
   - Config delta: replay the original winning confidence-gated dual-teacher recipe, but start directly from the current best `0.6339` checkpoint
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6300`
     - Conclusion: the old winning recipe does not reproduce its gain when started from the stronger checkpoint, so its earlier benefit was stage-specific.

35. `b7d20a7` `exp35: add confidence dual-teacher aux025 short`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_aux025_short.sh`
   - Config delta: keep the same replay setup as exp34 but reduce the auxiliary student-teacher logit weight from `0.4` to `0.25`
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6297`
     - Conclusion: lowering the self-teacher contribution alone does not recover the lost F1 on top of the `0.6339` checkpoint.

36. `ff8d8c3` `exp36: add confidence dual-teacher gentle polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_gentle_polish_short.sh`
   - Config delta: keep the confidence-gated dual-teacher family, but switch to a gentler one-epoch polish pass with lower LR, lower aux weight, and higher hard-label emphasis
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6851`, `macro_f1=0.6350`
     - New best experiment so far
     - Checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6851_f1_0.6350`
     - Conclusion: a conservative polish pass finally improved on the previous best, suggesting the student now benefits more from checkpoint-preserving refinement than stronger teacher pull.

37. `ee52e2f` `exp37: add minilm gentle polish distillation`
   - Script: `experiments/run_kd_laptop_minilm_gentle_polish_short.sh`
   - Config delta: start from the new best checkpoint, keep the gentle polish schedule, and add a very light MiniLM-style token relation loss
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6319`
     - Conclusion: even a very light relation objective is still too disruptive once the gentle-polish checkpoint is already strong.

38. `04dad33` `exp38: add confidence dual-teacher gentle polish replay`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_gentle_polish_replay_short.sh`
   - Config delta: replay the exp36 gentle-polish recipe directly from the new `0.6350` best checkpoint
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6316`
     - Conclusion: the exp36 gain does not stack through an immediate second identical pass; it appears to be a narrow early improvement rather than a repeatable iterative climb.

39. `e9d31d9` `exp39: add minilm dual-teacher aux025 polish`
   - Script: `experiments/run_kd_laptop_minilm_dual_teacher_aux025_short.sh`
   - Config delta: start from the `0.6350` best checkpoint, keep the gentle dual-teacher polish setup, reduce the auxiliary student-teacher weight to `0.25`, and add an ultra-light MiniLM-style token relation term
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6316`
     - Conclusion: even an almost negligible relation loss still slides back into the same lower-F1 band, so MiniLM-style structure transfer remains too disruptive at this stage.

40. `ef28306` `exp40: add teacher gentle polish distillation`
   - Script: `experiments/run_kd_laptop_teacher_gentle_polish_short.sh`
   - Config delta: keep the same gentle polish schedule but remove the auxiliary student teacher, leaving only the BERT teacher
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6788`, `macro_f1=0.6255`
     - Conclusion: the stronger late-stage checkpoint still benefits from the dual-teacher setup; teacher-only polishing loses too much F1.

41. `9f79197` `exp41: add ultra-gentle dual-teacher polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_ultra_gentle_polish_short.sh`
   - Config delta: continue from the `0.6350` best checkpoint, keep confidence-gated dual teachers, and make the polish pass even lighter by reducing KD emphasis further
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6804`, `macro_f1=0.6266`
     - Conclusion: pushing the recipe to be even gentler removes too much useful teacher signal and falls well short of the current best.

42. `72f69bd` `exp42: add lagged-aux dual-teacher polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_lagged_aux_polish_short.sh`
   - Config delta: initialize from the `0.6350` best checkpoint but switch the auxiliary student teacher back to the previous `0.6339` checkpoint to reintroduce a slightly older self-teacher signal
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6316`
     - Conclusion: the gain from exp36 was not caused by simply using a lagged self-teacher; replaying that idea from the stronger checkpoint still lands below the best.

43. `4ae94a9` `exp43: add dual-teacher gentle ramp polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_gentle_ramp2_short.sh`
   - Config delta: keep the current best checkpoint and dual-teacher setup, but reduce LR to `3e-5` and spread KD across a two-epoch cosine ramp instead of a one-epoch full-strength pass
   - Result:
     - Completed two short epochs and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6291`
     - Conclusion: stretching the polish over two gentler epochs suppresses the already narrow early peak rather than improving it.

44. `328a6e3` `exp44: add gentle primary-weight dual-teacher polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_primary_weight_gentle_short.sh`
   - Config delta: keep the gentle one-epoch dual-teacher pass, lower the aux blend slightly, and compute per-instance KD weights from the primary BERT teacher
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6835`, `macro_f1=0.6316`
     - Observed `kd_w_min < kd_w_max`, confirming primary-teacher weighting became active again
     - Conclusion: restoring meaningful instance weighting changes the optimization dynamics, but it still does not recover the `0.6350` peak.

45. `3fc656a` `exp45: add agreement-weighted gentle dual-teacher polish`
   - Script: `experiments/run_kd_laptop_dual_teacher_agreement_weighted_gentle_short.sh`
   - Config delta: add a new teacher-agreement weighting term in `train.py` so KD is emphasized more on samples where the BERT teacher and auxiliary student teacher agree
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6294`
     - Conclusion: filtering KD by teacher agreement is coherent and stable, but on this checkpoint it reduces useful signal more than harmful disagreement and performs worse than the simpler gentle recipe.

46. `1960bb2` `exp46: fix dual-teacher prob-blend targets`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_gentle_probfix_short.sh`
   - Config delta: change `train.py` so probability-space dual-teacher blending feeds the blended probabilities directly into the KL target instead of implicitly applying the KD temperature a second time
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6820`, `macro_f1=0.6258`
     - Conclusion: the previous probability-space recipe was relying on the extra target softening; removing it cleanly makes the dual-teacher pass substantially weaker.

47. `8079bba` `exp47: add blended-target temperature control`
   - Script: `experiments/run_kd_laptop_dual_teacher_confidence_gentle_blendedtemp2_short.sh`
   - Config delta: keep the new probability-target path but add an explicit `--kd_blended_target_temperature` and test an intermediate value `2.0`
   - Result:
     - Completed one short epoch and persisted the best checkpoint
     - Best observed checkpoint: `acc=0.6804`, `macro_f1=0.6266`
     - Conclusion: partially restoring the extra softening recovers a little from exp46, but it still remains far below the original `0.6350` best.

48. `3d27286` `exp48: add deeper warm-start student polish`
   - Script: `experiments/run_kd_laptop_deeper_student_warmstart_short.sh`
   - Config delta: increase the student to two LSTM layers while partially warm-starting compatible weights from the `0.6350` checkpoint
   - Result:
     - Early-stopped after severe collapse
     - Best observed checkpoint before stop: `acc=0.5396`, `macro_f1=0.3225`
     - Diagnostic: the auxiliary student teacher was unintentionally instantiated with the deeper two-layer architecture too, so the one-layer checkpoint only partially loaded into the aux teacher
     - Conclusion: the first deeper-student attempt was invalid as a fair dual-teacher test and clearly unstable.

49. `dd89516` `exp49: add fixed-aux deeper warm-start student`
   - Script: `experiments/run_kd_laptop_deeper_student_warmstart_fixed_aux_short.sh`
   - Config delta: add `--teacher_student_encoder_layers` so the auxiliary student teacher can stay one-layer while the student is trained as a two-layer model, then rerun the deeper warm-start experiment
   - Result:
     - Early-stopped after severe collapse
     - Best observed checkpoint before stop: `acc=0.5475`, `macro_f1=0.3225`
     - Conclusion: once the aux-teacher architecture mismatch is fixed, the deeper warm-start student still collapses badly, so late-stage depth expansion is not viable in the current optimization setup.

50. `c2ddda6` `exp50: add hidden40 expansion warm-start`
   - Script: `experiments/run_kd_laptop_wider_hidden40_expand_short.sh`
   - Config delta: add a structured student expansion loader in `train.py` and widen the student hidden size from `32` to `40` while overlap-copying the best checkpoint into the larger model
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.5854`, `macro_f1=0.5226`
     - Diagnostic: this first width-expansion run instantiated the auxiliary student teacher at `hidden_dim=40`, so the `hidden_dim=32` checkpoint only partially loaded into the aux teacher
     - Conclusion: the new expansion loader works for the student itself, but this first widened-student result is not a fair dual-teacher evaluation because the auxiliary teacher architecture was mismatched.

51. `106206e` `exp51: add fixed-aux hidden40 expansion warm-start`
   - Script: `experiments/run_kd_laptop_wider_hidden40_expand_fixed_aux_short.sh`
   - Config delta: add `--teacher_student_hidden_dim`, `--teacher_student_pos_dim`, and `--teacher_student_post_dim` so the auxiliary student teacher can keep the original `32/8/8` architecture while the student is widened to `40/8/8`
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.5649`, `macro_f1=0.5088`
     - Conclusion: once the auxiliary teacher width mismatch is fixed, the widened student still falls far below the baseline, so simple width expansion with overlap-copy initialization is not sufficient.

52. `8450546` `exp52: add hidden40 logit-only expansion warm-start`
   - Script: `experiments/run_kd_laptop_wider_hidden40_logits_expand_short.sh`
   - Config delta: keep the widened `hidden_dim=40` student with expansion init and the fixed-width auxiliary teacher, but nearly remove feature KD by setting `kd_gamma=0.0`
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.5570`, `macro_f1=0.5094`
     - Conclusion: the widened student remains poor even when feature KD is removed, so the failure is not just a feature-alignment issue; the broader width-expansion recipe itself is unstable.

53. `2241a8d` `exp53: add zero-preserve hidden40 expansion`
   - Script: `experiments/run_kd_laptop_wider_hidden40_zero_preserve_short.sh`
   - Config delta: keep the widened `hidden_dim=40` student and fixed-width auxiliary teacher, but change expansion initialization to `zero_preserve`, which zeroes new dimensions before copying the old checkpoint into the shared subspace
   - Result:
     - Early-stopped after clear underperformance
     - Best observed checkpoint before stop: `acc=0.5649`, `macro_f1=0.5328`
     - Diagnostic: `kd_feature` dropped sharply versus exp51/exp52, confirming the safer initialization made the widened student much closer to the original function at the start
     - Conclusion: even with near-function-preserving expansion, the widened student still lands far below the `0.6350` checkpoint, so late-stage width expansion is not a viable path here.

Current best experiment:
- Commit: `ff8d8c3`
- Script: `experiments/run_kd_laptop_dual_teacher_confidence_gentle_polish_short.sh`
- Best selected checkpoint: `state_dict/ssegcnbertstudent_laptop_acc_0.6851_f1_0.6350`

54. `8aa7b59` `exp54: add shallow bert kd student`
   - Script: `experiments/run_kd_laptop_shallowbert6_uniform_short.sh`
   - Config delta: add a new `ssegcnbertshallow` KD student that keeps the BERT+GCN ABSA head, truncates BERT to `6` layers, and initializes from the fine-tuned teacher by uniformly remapping teacher encoder layers `[0, 2, 4, 7, 9, 11]`
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7358_f1_0.6884`
     - Best observed accuracy during training: `acc=0.7421`
     - Best observed macro F1 during training/final selection: `macro_f1=0.6884`
   - Conclusion: this BERT-student pivot is materially stronger than the tiny word-side student (`0.6884` vs `0.6350` macro F1), so the new strongest line of work is shallow BERT distillation rather than continued polishing of `ssegcnbertstudent`.

55. `4216c4d` `exp55: tune shallow bert logit kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_last_logits_short.sh`
   - Config delta: continue directly from the new shallow-BERT best checkpoint, drop feature KD (`kd_gamma=0.0`), raise logit KD pressure (`kd_beta=0.45`, `temperature=3.0`), lower the BERT LR slightly, and keep the 6-layer student focused on its own copied top stack rather than reinitializing from the 12-layer teacher
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7484_f1_0.7078`
     - Best observed accuracy during training: `acc=0.7563`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7078`
   - Conclusion: for the shallow BERT student, logit-only KD is much better than pooled feature matching at this stage, and the student is now very close to the teacher's macro F1 while already exceeding the teacher's accuracy.

56. `e1fa9b3` `exp56: add shallow bert tinybert-lite kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_tinybertlite_short.sh`
   - Config delta: fix token-level KD support for `ssegcnbertshallow`, continue from the `exp55` best checkpoint, keep the strong logit-KD backbone, and add very light TinyBERT-style token hidden/relation distillation (`kd_token_hidden_weight=0.03`, `kd_token_relation_weight=0.01`) with a slightly lower BERT LR
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7595_f1_0.7100`
     - Best observed accuracy during training/final selection: `acc=0.7595`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7100`
   - Conclusion: light token-level distillation is helpful but only marginally so; it improves on the prior best (`0.7100` vs `0.7078`) and sets a new student peak, yet still remains just below the teacher's `0.7161` macro F1.

57. `9e40ec3` `exp57: tune shallow bert hidden-only kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_hiddenonly_short.sh`
   - Config delta: continue from the `exp56` best checkpoint, drop token-relation KD completely, keep only cosine token-hidden KD, reduce the BERT LR again, and slightly raise the token-hidden weight
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7595_f1_0.7153`
     - Best observed accuracy during training/final selection: `acc=0.7595`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7153`
   - Conclusion: relation KD had become net noise near the optimum; hidden-only token distillation is cleaner and pushes the student to within `0.0008` macro F1 of the teacher.

58. `7a51aed` `exp58: polish shallow bert hidden-only kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_hiddenonly_polish_short.sh`
   - Config delta: continue from the `exp57` best checkpoint, lower the BERT LR further for a gentle polish stage, and shift a bit more weight toward the hard target while keeping the same hidden-only token KD recipe
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7658_f1_0.7260`
     - Best observed accuracy during training/final selection: `acc=0.7658`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7260`
   - Conclusion: this continuation finally surpasses the teacher on macro F1 (`0.7260` vs `0.7161`) while also lifting accuracy to `0.7658`; the shallow BERT student now beats the teacher on the primary F1 metric, though it still trails slightly on accuracy.

59. `b2e9d76` `exp59: push shallow bert hard-target balance`
   - Script: `experiments/run_kd_laptop_shallowbert6_hiddenonly_accuracy_push_short.sh`
   - Config delta: continue from the `exp58` best checkpoint, lower the BERT LR again, increase hard-target emphasis (`kd_alpha=0.66`), reduce logit KD weight slightly, and lighten the token-hidden term a bit to bias toward accuracy recovery
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7658_f1_0.7275`
     - Best observed accuracy during training: `acc=0.7674`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7275`
   - Conclusion: shifting slightly toward the hard labels improves macro F1 again and nudges the best observed student accuracy upward, but still does not overtake the teacher's `0.7722` accuracy.

60. `75d896e` `exp60: edge shallow bert toward hard labels`
   - Script: `experiments/run_kd_laptop_shallowbert6_hardlabel_edge_short.sh`
   - Config delta: continue from the `exp59` best checkpoint, push further toward hard-label fitting (`kd_alpha=0.70`, `kd_beta=0.25`), reduce token-hidden KD to a very light regularizer, and lower the BERT LR once more
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7674_f1_0.7278`
     - Best observed accuracy during training/final selection: `acc=0.7674`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7278`
   - Conclusion: the student keeps improving on both headline metrics versus the previous best and remains clearly above the teacher on macro F1, but the remaining accuracy gap persists even under stronger hard-label pressure.

61. `9764e9f` `exp61: sweep shallow bert hard-label polish`
   - Scripts:
     - `experiments/run_kd_laptop_shallowbert6_hardlabel_edge2_short.sh`
     - `experiments/run_kd_laptop_shallowbert6_hiddenlite_polish_short.sh`
   - Config delta: sweep two very local continuations from the `exp60` best checkpoint with even lower BERT LR and lighter token-hidden KD; the winning branch pushes hard-label emphasis slightly further (`kd_alpha=0.72`, `kd_beta=0.23`, `kd_token_hidden_weight=0.015`)
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7706_f1_0.7317`
     - Best observed accuracy during training/final selection: `acc=0.7706`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7317`
   - Conclusion: the local hard-label polish still had headroom; the student now comes within `0.0016` accuracy of the teacher while extending its macro-F1 lead further.

62. `5058a92` `exp62: probe shallow bert accuracy margin`
   - Script: `experiments/run_kd_laptop_shallowbert6_accuracy_margin_short.sh`
   - Config delta: continue from the `exp61` best checkpoint, ease hard-label pressure slightly (`kd_alpha=0.74`, `kd_beta=0.21`), reduce the token-hidden regularizer to `0.01`, and raise the BERT LR a bit to test whether a slightly softer continuation can recover the last accuracy gap without giving up the student F1 edge
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7706_f1_0.7302`
     - Best observed accuracy during training/final selection: `acc=0.7706`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7302`
   - Conclusion: softening this local continuation does not improve on `exp61`; it matches the same best observed accuracy but loses macro F1, so the strongest branch remains the harder-label `exp61` checkpoint.

63. `afaf02b` `exp63: probe shallow bert dkd last mile`
   - Script: `experiments/run_kd_laptop_shallowbert6_dkd_lastmile_short.sh`
   - Config delta: continue from the `exp61` best checkpoint, replace plain KL logit KD with DKD (`kd_logit_mode=dkd`) while raising hard-label weight to `0.76`, reducing KD weight to `0.18`, and keeping a very light cosine token-hidden regularizer to test whether decoupled target/non-target supervision can recover the last accuracy gap
   - Result:
     - Best selected checkpoint before failure: `state_dict/ssegcnbertshallow_laptop_acc_0.7611_f1_0.7203`
     - Best observed accuracy before failure: `acc=0.7611`
     - Best observed macro F1 before failure: `macro_f1=0.7203`
   - Conclusion: this local DKD continuation is materially worse than the `exp61` optimum even before completion; it also exposed an environment issue where `torch.save` failed because the root filesystem was full, so the run was stopped, cache space was reclaimed, and the strongest branch remains `exp61`.

64. `97b5e8e` `exp64: probe shallow bert dist last mile`
   - Script: `experiments/run_kd_laptop_shallowbert6_dist_lastmile_short.sh`
   - Config delta: continue from the `exp61` best checkpoint, replace plain KL logit KD with DIST (`kd_logit_mode=dist`) to preserve inter-sample and intra-class correlation structure, keep hard-label pressure high (`kd_alpha=0.76`), reduce KD weight to `0.12`, and retain a very light cosine token-hidden regularizer
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7737_f1_0.7360`
     - Best observed accuracy during training/final selection: `acc=0.7737`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7360`
   - Conclusion: this relation-aware logit distillation unlocks a new regime; the shallow BERT student now surpasses the teacher on both headline metrics, beating teacher accuracy by `0.0015` and extending the macro-F1 lead to `0.0199`.

65. `bd9a51f` `exp65: push shallow bert dist polish`
   - Script: `experiments/run_kd_laptop_shallowbert6_dist_push_short.sh`
   - Config delta: continue from the new `exp64` best checkpoint, lower the BERT LR to `2e-6`, raise hard-label weight to `0.78`, reduce DIST weight slightly (`kd_beta=0.10`), emphasize intra-class structure more (`kd_dist_intra_weight=2.5`), and lighten token-hidden KD to `0.008` for a conservative polish stage
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7706_f1_0.7318`
     - Best observed accuracy during training/final selection: `acc=0.7706`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7318`
   - Conclusion: once the DIST branch crossed into the new optimum, extra hard-label pressure and lower LR only pulled it back toward the old plateau; `exp64` remains the strongest checkpoint by a clear margin.

66. `8bfed48` `exp66: add shallow bert dist patient kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_dist_patient_short.sh`
   - Config delta: add PatientKD-style intermediate BERT hidden-layer distillation support, then continue from the `exp64` best checkpoint with the same winning DIST setup while adding a light hidden-layer cosine loss (`kd_hidden_layer_weight=0.02`, `kd_hidden_layer_map=last`) to test whether multi-layer supervision can improve the already-strong shallow BERT student
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7706_f1_0.7323`
     - Best observed accuracy during training/final selection: `acc=0.7706`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7323`
   - Conclusion: this first PatientKD-style hidden-layer continuation does not beat the simpler DIST optimum; additional intermediate supervision appears to over-constrain the local best region, so `exp64` remains the strongest model.

67. `b60cba5` `exp67: ablate shallow bert dist token kd`
   - Script: `experiments/run_kd_laptop_shallowbert6_dist_notoken_short.sh`
   - Config delta: continue from the `exp64` best checkpoint, keep the winning DIST setup intact, but nearly remove token-hidden KD (`kd_token_hidden_weight=0.002`) to test whether the new optimum is now dominated by relation-aware logit KD alone rather than by residual token-level regularization
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7722_f1_0.7335`
     - Best observed accuracy during training/final selection: `acc=0.7722`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7335`
   - Conclusion: stripping token-level KD almost entirely lets the student recover teacher-level accuracy again, but still does not beat the `exp64` optimum; a small amount of token-hidden regularization appears to remain helpful in the best DIST regime.

68. `4bfba73` `exp68: balance shallow bert dist terms`
   - Script: `experiments/run_kd_laptop_shallowbert6_dist_balance_short.sh`
   - Config delta: continue from the `exp64` best checkpoint, keep the same DIST-based recipe family, but rebalance the relation-aware logit terms toward a more even inter/intra split (`kd_dist_inter_weight=1.4`, `kd_dist_intra_weight=1.6`), lower `kd_beta` slightly to `0.11`, and trim token-hidden KD a bit to test whether the `exp64` optimum can be sharpened rather than merely reproduced
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7737_f1_0.7360`
     - Best observed accuracy during training/final selection: `acc=0.7737`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7360`
   - Conclusion: this balanced DIST variant successfully reproduces the current best student but does not improve on it; the `exp64` recipe remains the strongest known checkpoint, and the search region around it appears very flat.

69. `eaebb87` `exp69: recreate local restaurant teacher baseline`
   - Script: `experiments/run_teacher_restaurant_short.sh`
   - Config delta: regenerate `dataset/Restaurants_corenlp/train_write.json` so the restaurant train split matches the full 1980 examples again, then run the local one-epoch `ssegcnbert` teacher recipe directly on that repaired dataset to produce a distillation teacher without spending cycles on a stronger paper-style teacher rebuild
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbert_restaurant_acc_0.8365_f1_0.7439`
     - Best observed accuracy during training/final selection: `acc=0.8365`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7439`
   - Conclusion: the repaired local restaurant teacher is now reproducible and ready for cross-dataset validation of the `exp64` DIST recipe; its headline metrics remain below the earlier paper-level teacher numbers, but it satisfies the current constraint to distill directly from the available local teacher.

70. `1132384` `exp70: add cross-dataset exp64 validation chain`
   - Scripts:
     - `experiments/run_kd_shallowbert6_exp64_chain_short.sh`
     - `experiments/run_kd_restaurant_shallowbert6_dist_short.sh`
     - `experiments/run_kd_twitter_shallowbert6_dist_short.sh`
     - `experiments/run_teacher_twitter_short.sh`
   - Config delta: convert the laptop-only `exp64` win into a reusable multi-stage chain that faithfully reproduces the actual warm-start protocol behind `exp54 -> exp61 -> exp64`, then add the missing local twitter teacher recipe and dataset-specific wrappers so any student that matches or beats its teacher can now be validated across all three datasets under the same method family
   - Result:
     - Infrastructure commit only; no student metrics attached yet
   - Conclusion: the repository now has the correct control scripts to validate the winning shallow-BERT DIST method across `laptop`, `restaurant`, and `twitter` without falsely treating a DIST-from-scratch run as equivalent to the true `exp64` continuation protocol.

71. `cf52486` `exp71: recreate local twitter teacher baseline`
   - Script: `experiments/run_teacher_twitter_short.sh`
   - Config delta: run the local one-epoch `ssegcnbert` teacher recipe directly on `Tweets_corenlp` so the cross-dataset validation chain has a matching local twitter teacher checkpoint instead of relying on a missing or paper-only teacher artifact
   - Result:
     - Best selected checkpoint: `state_dict/ssegcnbert_twitter_acc_0.7386_f1_0.7317`
     - Best observed accuracy during training/final selection: `acc=0.7386`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7317`
   - Conclusion: the local twitter teacher is now reproducible and substantially stronger than the earlier short-run intermediates, giving the `exp64` student chain a realistic cross-dataset validation target on twitter.

72. `9d36844` `exp72: validate exp64 chain on twitter`
   - Script: `experiments/run_kd_twitter_shallowbert6_dist_short.sh`
   - Config delta: run the full shallow-BERT continuation chain behind `exp54 -> exp61 -> exp64` on `twitter`, using the local twitter teacher and preserving the winning protocol rather than treating DIST-from-scratch as equivalent to the laptop result
   - Result:
     - Best selected checkpoint across the twitter chain: `state_dict/ssegcnbertshallow_twitter_acc_0.7710_f1_0.7589`
     - Best observed accuracy during training/final selection: `acc=0.7710`
     - Best observed macro F1 during training/final selection: `macro_f1=0.7589`
     - Teacher reference: `state_dict/ssegcnbert_twitter_acc_0.7386_f1_0.7317`
   - Conclusion: the same shallow-BERT continuation family that won on laptop transfers successfully to twitter as well, beating the local teacher by `0.0324` accuracy and `0.0272` macro F1.

Current best experiment:
- Commit: `97b5e8e`
- Script: `experiments/run_kd_laptop_shallowbert6_dist_lastmile_short.sh`
- Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7737_f1_0.7360`
