# NoPost Feature Lowband Family Summary

Date: 2026-07-04

Status: WLDB-A, the tested v2.18/v2.19/v2.20 deployable lowband predictor
forms, the v2.23 small-adapter train form, the v2.25A direct risk soft-label /
scale-distillation form, and the v2.26 current-risk-input trainability route
are not training-authorized. NoPost lowband remains directionally open only for
a materially new route that fixes safety/no-op calibration, trained-gate/canary
collapse, and tail preservation before training.

## Sources

- Index: `../EXPERIMENT_INDEX.md`
- Cards:
  - `../experiment_cards/2026-07-03-haze4k-v2-16-nopost-wavelet-lowband-decoder.md`
  - `../experiment_cards/2026-07-03-haze4k-v2-17-nopost-lowband-alignment-tail-audit.md`
  - `../experiment_cards/2026-07-03-haze4k-v2-18-nopost-tailaware-lowband-policy.md`
  - `../experiment_cards/2026-07-03-haze4k-v2-20-nopost-midfinal-context-lowband-learnability.md`
  - `../experiment_cards/2026-07-04-haze4k-v2-24-nopost-train-time-controller-failure-audit.md`
  - `../experiment_cards/2026-07-04-haze4k-v2-25a-nopost-risk-softlabel-scale-distill.md`
  - `../experiment_cards/2026-07-04-haze4k-v2-26-nopost-risk-signal-separability-audit.md`
- Evidence roots:
  - `../experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/`
  - `../experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/`
  - `../experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703/`
  - `../experiment_logs/haze4k_v2_19_nopost_spatial_lowband_policy_learnability_20260703/`
  - `../experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703/`
  - `../experiment_logs/haze4k_v2_24_nopost_train_time_controller_failure_audit_20260704/`
  - `../experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704/`
  - `../experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/`

## Established Facts

| Route | Main result | Decision |
| --- | --- | --- |
| Haze4K v2.16 NoPost Wavelet Lowband Decoder | T0 showed WD0375 severe-risk is decoupled from lowband need: WD0375 severe vs lowband-need Jaccard `0.027917`; WD0375 severe vs A0 hard-bottom25 Jaccard `0.000000`. T1 LL oracle was very strong: all-image mean `+14.998694`, hard `+18.939359`, easy `+11.853745`, severe `0`, lowband-need rate `1.000000`. T2 source/identity passed with forbidden symbol hits `0` and identity max abs diff `1.7881393432617188e-07`. WLDB-A then trained seed `3407` for `20` epochs with only `2128` trainable `nopost_wldb.*` params. Best checkpoint `model_5` had mean/hard/easy `+0.081889/+0.105887/+0.020994`, positive `0.662500`, but severe `67/480` and strong-reference regressions `48/120`. | `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`; do not expand WLDB-A seeds, epochs, hidden width, locked test, or promotion from this form. |
| Haze4K v2.17 NoPost Lowband Alignment Tail Audit | R1 confirmed the v2.16 failure shape: `model_5` mean/hard/easy `+0.081889/+0.105887/+0.020994`, p05 `-0.438669`, severe `67/480`, action-budget activation `0`. R2 proved internal feature-lowband oracle headroom: O1 global final-feature LL mean/hard/easy `+0.842954/+1.591207/+0.359026`, p05 `+0.001803`, severe `0`; O2 spatial final mean `+6.160490`; O3 mid+final mean `+6.832469`; O4 RGB LL reference mean `+14.998694`. R3 showed the average objective improved final/lowband L1 but failed CVaR/tail safety: CVaR5 `-0.646619`, severe `67/480`, strong-reference regressions `48/120`, budget activation `0.0`. | `NO_TRAINING_PAUSE_DESIGN_TAIL_AWARE_WLDB_A2_OR_WLDB_B_OBJECTIVE`; close WLDB-A as trained, keep NoPost lowband open, require a materially changed tail-aware objective and gate before training. |
| Haze4K v2.18 NoPost Tail-Aware Lowband Policy | P4 passed after an engineering broadcast fix: forward `(self, x)`, forbidden symbol hits `0`, zero-init max_abs_vs_A0 `0.0` for global and spatial modes, new params global `3168` and spatial `19552`, official params `8630665`. P1 regenerated O1 global final-feature LL oracle actions and tested deployable pooled-LL MLP learnability. It beat shuffled control and moved hard samples, but was not tail-safe: mean/hard/easy `+0.263178/+0.859418/-0.183929`, p05 `-1.164642`, CVaR5 `-2.050251`, severe `568/2400`, strong-reference regressions `303/600`, positive `0.592083`, wrong-direction `0.193750`, control gap vs shuffled `+0.308958`. P2 passed: tail/preserve objective replay covered v2.16 `model_5` severe and strong/easy failures with positive activation rates `0.0`. P3 passed: `3` predeclared nonzero action-budget thresholds were calibrated. | `V218_PAUSE_P1_GLOBAL_POLICY_LEARNABILITY_FAIL`; do not train WLDB-A2 global pooled final-feature LL policy. Use P2/P3/P4 as positive evidence for a future spatial WLDB-B learnability route. |
| Haze4K v2.19 NoPost Spatial Lowband Policy Learnability | P0 passed source-clean/identity for the O2 final-only spatial NoPost policy. P1 best small-CNN spatial predictor had mean `+0.9921`, hard `+2.6504`, and positive ratio `0.7192`, but failed safety with easy `-0.1346`, p05 `-1.1486`, CVaR5 `-2.1058`, severe rate `0.2025`, strong-reference regressions `302/600`, and fold tail pass `0/5`. P3 objective replay covered failures but remained guard evidence only. | `V219_LEARNABILITY_FAIL_OR_GUARD_FAIL_PAUSE_BEFORE_TRAINING`; do not train WLDB-B from the current O2 final-only spatial predictor form. |
| Haze4K v2.20 NoPost Mid+Final Context Lowband Learnability | P0 passed source-clean/identity and official partial load. P1-A mechanism passed for `P1_final_mid_global_context_predictor`: mean `+2.0684`, hard `+4.1450`, easy `+0.5199`, positive ratio `0.8508`, wrong-direction `0.00417`, and real-vs-shuffled gap `+3.1959`. P1-B failed safety with p05 `-0.7255`, CVaR5 `-1.6967`, severe rate `0.11125`, strong-reference regression rate `0.2667`, and fold tail pass `0/5`. P2 found useful unsafe-action/no-op classifier signal, P3 showed remaining tail damage is not explained mainly by direction/peak alone, and P4 passed objective replay as a guard only. No training launched and locked Haze4K was untouched. | `V220_P1A_PASS_P1B_FAIL_NORMAL_GATE_PAUSE_NO_TRAINING`; do not train this O3 mid+final/global-context predictor. |
| Haze4K v2.24 NoPost Train-Time Controller Failure Audit | Diagnostic P0-P5 completed on `convir-4090` after v2.23 OOF failure. P2 confirmed train-time risk-head collapse: trained risk probability stayed near initialization (`mean 0.1822009`, `std 0.0005259`, ROC-AUC `0.5175`) while v2.21 replay probability on the same crop set retained ROC-AUC `0.9149`. P3 showed trained action can be rescued by oracle unsafe gating (mean `+0.1177 dB`, severe `0`, strong regressions `0`), but v2.21 replay gate alone did not pass. P4 recorded supervision/gradient imbalance risk, and P5 showed epoch2 mean can rise while tail worsens. | `V224_DIAGNOSTIC_COMPLETE_CASE_A_RISK_HEAD_COLLAPSE_LOCKED_TEST_BLOCKED`; do not expand v2.23. Only a trained-gate calibration route was authorized, and locked test stayed untouched. |
| Haze4K v2.25A NoPost Risk Soft-Label / Scale Distillation | Risk/context-only calibration screen completed with action heads frozen and locked test untouched. Direct distillation from v2.21 `unsafe_action_probability` and `risk_scale` failed the predeclared gate: probability std `0.001669`, ROC-AUC `0.5501`, AP `0.4938`, ECE10 `0.0526`, target probability MAE `0.2488`. Each fold produced constant validation probability (`trained_prob_std=0.0`). | `V225A_RISK_CALIBRATION_GATE_FAIL_NORMAL_PAUSE`; stop v2.25A. Do not launch post-train factorial rescue, action joint training, or locked-test evaluation from this result. |
| Haze4K v2.26 NoPost Risk Signal Separability Audit | P0 showed the v2.25A tuple-sort AP was an invalid diagnostic artifact: old AP `0.4938`, tie-aware AP `0.1397` near base rate `0.1271`; the v2.25A fail remains by probability std `0.001669`, ROC-AUC `0.5501`, and target MAE `0.2488`. P1 replay passed with `0` missing joins and v2.21 scalar MLP replay AUC `0.9286`, AP `0.6995`. P2 found current risk features weak/inconclusive: positive-control v2.21 cached scalar MLP AUC `0.9603`, AP `0.7788`, while best current feature `B_final_ll_pooled` had AUC `0.6436`, AP `0.2061`. P3 canary32/canary64 both failed at train AUC `0.5`, prob std `0.0`, and P4 did not minimally rescue trainability (`best val_auc=0.7407`, prob std `0.0224`). | `V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED`; do not continue rescue, action joint training, or locked Haze4K from this route. |

## Family Verdict

The precise conclusion is: close WLDB-A, v2.18 WLDB-A2 global pooled policy,
v2.19 O2 final-only spatial predictor, v2.20 O3 mid+final/global-context
predictor, v2.23 small-adapter train form, v2.25A direct risk soft-label /
scale-distillation, and v2.26 current-risk-input calibration/trainability as
training candidates. Do not close the underlying lowband-capacity direction,
but treat it as requiring a materially new safety/no-op/tail route and a new
trained-gate mechanism that first explains the constant-probability/canary
collapse.

v2.16 established that lowband correction is a real source of headroom inside
ConvIR-B. The RGB LL oracle and the proposed zero-init WLDB insertion were both
source-clean enough to justify a first trainable screen. That screen is now
closed because the concrete WLDB-A form only moved mean/hard metrics modestly
and failed tail safety. The severe failures are not a locked-test artifact:
they are train-derived fold evidence and no WLDB-A checkpoint passed the
predeclared gate.

v2.17 explains why the direction should not be closed. Internal feature-lowband
oracles are strong, especially spatial and mid+final insertion oracles. The
WLDB-A failure is better explained by objective and constraint mismatch than by
absence of lowband capacity. The current average objective can improve L1 while
still leaving p05/CVaR/severe and strong/easy preservation failures, and the
existing action-budget term did not activate.

v2.18 tested the recommended next filter before training. The contract and
identity checks passed, the tail/preserve replay catches the known WLDB-A
failure mode, and a nonzero action budget can be calibrated. The blocker was
P1: the deployable O1 global pooled final-feature LL policy learned average and
hard movement but damaged easy and tail cases too severely.

v2.19 then tested the spatial O2 final-only form suggested by the v2.17 oracle
gap. It improved mechanism strength over v2.18, but the same safety shape
remained: easy, p05, CVaR5, severe, strong-reference, and fold-tail gates did
not authorize training.

v2.20 tested the stronger O3 mid+final/global-context form. This was a genuine
mechanism improvement over v2.19, including much larger mean and hard movement
and very low wrong-direction rate. It still failed the P1-B training safety
gate, with fold tail pass `0/5` and substantial p05/CVaR/severe/strong-reference
risk. P2/P3/P4 are useful diagnostics and guard evidence, not training
authorization.

v2.24, v2.25A, and v2.26 explain why simple continuation after v2.23 is not
valid. v2.24 located the failure in replay-to-train safety controller transfer:
the trained gate collapsed to near-base-rate probabilities even though replay
probability remained highly ranked. v2.25A then tested the natural repair,
direct risk soft-label / scale distillation, and it still produced constant
fold probabilities. v2.26 invalidated the old tuple-sort AP comfort signal,
confirmed the v2.21 replay join and positive-control separability, then showed
the current risk inputs are weak and the canary/minimal optimizer trainability
checks fail. AP and ECE partial passes are not enough when probability spread,
ROC-AUC, target-probability MAE, canary overfit, and minimal optimizer checks
fail.

## Do Not Repeat Without New Evidence

- Do not expand WLDB-A with more seeds, epochs, hidden width, checkpoint
  selection, or locked-test use.
- Do not train WLDB-A2 global pooled final-feature LL policy from the current
  P1 result.
- Do not train the current v2.19 O2 final-only spatial predictor form.
- Do not train the current v2.20 O3 mid+final/global-context predictor form.
- Do not continue v2.23, v2.25A, or v2.26 with more epochs, more samples,
  broader folds, direct loss-weight tuning, action-joint training, rescue
  sweeps, or relaxed OOF gates.
- Do not treat the v2.21 replay pass, v2.25A AP/ECE partial passes, or the
  invalid tuple-sort AP as train-time success.
- Do not treat mean or hard-bucket improvement as sufficient if p05, CVaR,
  severe count, strong-reference regressions, or easy preservation fail.
- Do not use locked Haze4K feedback to tune lowband actions, objectives,
  thresholds, checkpoints, or route choice.
- Do not reopen the older NoPost severe-risk selector line unless it introduces
  a materially new signal beyond the v2.13-v2.15 failures.

## Reopen Condition

A credible follow-up must be a new safety-first NoPost lowband route, not a
WLDB-A rerun, a direct v2.19/v2.20 training launch, a v2.23 expansion, a v2.25A
hyperparameter extension, or a v2.26 rescue/action-joint continuation. It
should use the v2.17 O2/O3 oracle headroom and the v2.20 mechanism-positive
context result as capacity evidence, keep the source-clean and identity
contract requirements, and make trained-gate/canary trainability, no-op
calibration, p05/CVaR/severe, strong/easy preservation, fold-tail consistency,
and action-budget behavior primary gates before any training promotion or
locked-test request.
