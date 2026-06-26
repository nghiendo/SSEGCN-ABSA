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

Current best experiment:
- Commit: `8aa7b59`
- Script: `experiments/run_kd_laptop_shallowbert6_uniform_short.sh`
- Best selected checkpoint: `state_dict/ssegcnbertshallow_laptop_acc_0.7358_f1_0.6884`
