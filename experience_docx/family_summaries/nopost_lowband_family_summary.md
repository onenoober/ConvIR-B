# NoPost Feature Lowband Family Summary

Date: 2026-07-05

Status: WLDB-A, the tested v2.18/v2.19/v2.20 deployable lowband predictor
forms, the v2.23 small-adapter train form, the v2.25A direct risk soft-label /
scale-distillation form, the v2.26 current-risk-input trainability route, the
v2.27 same-sample ILFRB-ACS action-bank construction, the first v2.28 OOF
prototype bank, and the v2.29 safe-envelope / GT-free table-policy bank are not
training-authorized. NoPost lowband remains directionally open only for a safer
OOF/prototype action-bank route that preserves no-op/useful-action separation
while solving hard-to-easy cross-bucket leakage and GT-free table selection
before selector probes or training.

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
  - `../experiment_cards/2026-07-05-haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill.md`
  - `../experiment_cards/2026-07-05-haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit.md`
  - `../experiment_cards/2026-07-05-haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration.md`
- Evidence roots:
  - `../experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/`
  - `../experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/`
  - `../experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703/`
  - `../experiment_logs/haze4k_v2_19_nopost_spatial_lowband_policy_learnability_20260703/`
  - `../experiment_logs/haze4k_v2_20_nopost_midfinal_context_lowband_learnability_20260703/`
  - `../experiment_logs/haze4k_v2_24_nopost_train_time_controller_failure_audit_20260704/`
  - `../experiment_logs/haze4k_v2_25a_nopost_risk_softlabel_scale_distill_20260704/`
  - `../experiment_logs/haze4k_v2_26_nopost_risk_signal_separability_audit_20260704/`
  - `../experiment_logs/haze4k_v2_27_nopost_ilfrb_action_conditioned_selective_distill_20260705/`
  - `../experiment_logs/haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705/`
  - `../experiment_logs/haze4k_v2_29_nopost_ilfrb_acs_safe_oof_action_bank_calibration_20260705/`

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
| Haze4K v2.26 NoPost Risk Signal Separability Audit | P0 showed the v2.25A tuple-sort AP was an invalid diagnostic artifact: old AP `0.4938`, tie-aware AP `0.1397` near base rate `0.1271`; the v2.25A fail remains by probability std `0.001669`, ROC-AUC `0.5501`, and target MAE `0.2488`. P1 replay passed with `0` missing joins and v2.21 scalar MLP replay AUC `0.9286`, AP `0.6995`. P2 found current risk features weak/inconclusive: positive-control v2.21 cached scalar MLP AUC `0.9603`, AP `0.7788`, while best current feature `B_final_ll_pooled` had AUC `0.6436`, AP `0.2061`. P3 canary32/canary64 both failed at train AUC `0.5`, prob std `0.0`, and P4 did not minimally rescue trainability (`best val_auc=0.7407`, prob std `0.0224`). Supplemental correctness evidence confirmed all v2.25A fold checkpoints strictly loaded and target-key/fallback audits were clean, while P4 replay still failed. | `V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED`; do not continue rescue, action joint training, or locked Haze4K from this route. |
| Haze4K v2.27 NoPost ILFRB-ACS | P0 passed source/contract/identity after eval zero-mixture bypass: strict partial load clean, forbidden hits `0`, identity max abs `0.0`. P1 showed very strong integrated multi-stage feature-lowband capacity on the train-derived screen: `S6_early_mid_final` mean/hard/easy `+7.8509/+9.4244/+6.1829`, p05 `+4.5170`, CVaR5 `+3.8851`, severe `0`, strong-reference regressions `0`; `S5_bottleneck_mid` also passed. P2 failed action-bank stratification: no-op conservative preference `0/80`, hard medium/strong preference `1.0`, and strong unsafe rate `0.0`, leaving no deployable no-op/unsafe separation for P3. P3/P4/P5/P6 were not launched, training was not launched, and locked test was untouched. | `P2_FAIL_ACTION_BANK_STRATIFICATION_PAUSE`; do not train or probe selectors from this bank. Redesign P2 action construction/no-op pressure before any selector or microfit. |
| Haze4K v2.28 NoPost ILFRB-ACS Action-Bank Stratification Audit | P0 passed architecture-delta audit with no model-structure change from v2.27. P2A OOF prototype replay produced the missing stratification signal: conservative no-op preference `0.225`, easy top25 no-op/mild `0.40`, hard bottom25 medium/strong `0.75`, and fold-tail pass `5/5`. Selected deployable OOF prototypes had mean/hard/easy `+1.1377/+1.3769/+0.7035`, p05 `0.0`, CVaR5 `0.0`, severe `0`, and strong-reference regression rate `0.0`. The route still failed because diagnostic negative-control unsafe rate was `0.5504`, above the allowed `0.40` upper bound. P2B was skipped, training was not launched, and locked test was untouched. | `P2A_FAIL_OOF_ACTION_BANK_STRATIFICATION_PAUSE`; do not train selector or touch locked test. Follow-up must reduce unsafe negative-control exposure while preserving OOF no-op/useful-action separation. |
| Haze4K v2.29 NoPost ILFRB-ACS Safe OOF Action-Bank Calibration | P0 passed architecture-delta audit with no model-structure change from v2.28. P2A safe-envelope calibration found `bucket_strength_grid` as the best selected policy: mean/hard/easy `+0.8332/+1.3753/+0.3342`, p05 `0.0`, CVaR5 `0.0`, severe `0`, no-op rate `0.2375`, easy no-op/mild `1.0`, hard medium/strong `0.7`, fold-tail pass `5/5`. It still failed plausible safety controls: deployable mild unsafe `0.215625`, cross-bucket unsafe `0.4556`, hard-to-easy cross severe `0.7167`, and overstrong 1.5 unsafe `0.4417`. The best GT-free table policy `energy_norm_plus_bucket_strength` was too weak: mean/hard/easy `+0.0775/+0.2403/+0.0372`, p05 `-0.2199`, CVaR5 `-0.4333`, severe `0.0625`, fold-tail pass `3/5`, table pass `0`. P2B was skipped, training was not launched, and locked test was untouched. | `P2A_FAIL_SAFE_OOF_ACTION_BANK_CALIBRATION_PAUSE`; do not probe selector, train, or touch locked test. Follow-up must solve hard-to-easy cross-bucket leakage and GT-free table selection first. |

## Family Verdict

The precise conclusion is: close WLDB-A, v2.18 WLDB-A2 global pooled policy,
v2.19 O2 final-only spatial predictor, v2.20 O3 mid+final/global-context
predictor, v2.23 small-adapter train form, v2.25A direct risk soft-label /
scale-distillation, v2.26 current-risk-input calibration/trainability, and the
current v2.27 oracle-derived action-bank construction, the first v2.28 OOF
prototype bank, and the v2.29 safe-envelope / GT-free table-policy bank as
training candidates. Do not close the underlying lowband-capacity direction,
but treat it as requiring a safer OOF/prototype bank with deployable GT-free
selection before any selector or trainability probe.

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
checks fail. Supplemental v2.26 manifests closed the main implementation
questions: the v2.25A fold checkpoints loaded cleanly, target keys were present,
scale fallbacks were unused, epsilon-tie sanity did not rescue the diagnosis,
and all-variant P4 replay still missed the pass line. AP and ECE partial passes
are not enough when probability spread, ROC-AUC, target-probability MAE, canary
overfit, and minimal optimizer checks fail.

v2.27 tested the recommended architecture pivot: integrated internal
low-frequency restoration plus a no-op/mild/medium/strong action bank. It
confirmed the capacity half of the hypothesis even more strongly than prior
screens, but failed before selector work because the current oracle-derived
action bank is not stratified: no-op is never preferred and strong action is
never unsafe on the screen. That is a different and useful failure from v2.26,
but it still blocks P3/P4/P5 and any training.

v2.28 tested the correct next question: whether v2.27's same-sample oracle
deltas can become deployable out-of-fold prototypes. This was a useful partial
positive: no-op preference became nonzero, hard samples still preferred stronger
actions, easy samples reached the no-op/mild floor, and selected OOF prototypes
had positive mean/hard/easy with clean p05/CVaR/severe metrics. The route still
paused because diagnostic negative controls were too unsafe overall. That means
the next design should narrow or regularize prototype exposure, not train a
selector on the current unsafe bank.

v2.29 narrowed that question into safe-envelope calibration and GT-free table
selection. Bucket-aware strength helped: the best selected policy kept positive
mean/hard/easy movement, clean selected p05/CVaR/severe, and the v2.28
no-op/useful-action shape. It still did not pass because plausible
miscalibration remained unsafe, especially hard-to-easy cross-bucket leakage
and overstrong exposure. The GT-free table policy was much weaker than the
selected-policy upper bound and failed p05/CVaR/severe/fold-tail gates. This is
therefore a useful normal pause, not authorization to train a selector.

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
- Do not continue the current v2.27 bank by training a selector anyway; P2 must
  first show no-op/unsafe action stratification.
- Do not continue the current v2.28 bank by training a selector anyway; P2A
  failed because diagnostic negative-control unsafe rate exceeded the safety
  bound.
- Do not continue the current v2.29 bank by probing a selector, training, or
  relaxing gates; P2A still fails plausible cross-bucket and overstrong safety
  controls, and the GT-free table policy is not deployable.
- Do not treat the v2.21 replay pass, v2.25A AP/ECE partial passes, or the
  invalid tuple-sort AP as train-time success.
- Do not treat mean or hard-bucket improvement as sufficient if p05, CVaR,
  severe count, strong-reference regressions, or easy preservation fail.
- Do not use locked Haze4K feedback to tune lowband actions, objectives,
  thresholds, checkpoints, or route choice.
- Do not reopen the older NoPost severe-risk selector line unless it introduces
  a materially new signal beyond the v2.13-v2.15 failures.

## Reopen Condition

A credible follow-up must be a new safety-first NoPost lowband action-bank
route, not a WLDB-A rerun, a direct v2.19/v2.20 training launch, a v2.23
expansion, a v2.25A hyperparameter extension, a v2.26 rescue/action-joint
continuation, a v2.27 selector launch from the same-sample bank, or a v2.28
selector launch from the current unsafe OOF prototype bank, or a v2.29 selector
launch from the failed safe-envelope/table-policy bank. It should use the v2.17
O2/O3, v2.20, v2.27 P1, v2.28 OOF stratification, and v2.29 bucket-aware
strength evidence, keep the source-clean and identity contract requirements,
preserve no-op/useful-action strata, and make hard-to-easy cross-bucket safety
plus GT-free table-policy utility the first hard gates before trained-gate,
canary, risk-coverage, selector, training, or locked-test work.
