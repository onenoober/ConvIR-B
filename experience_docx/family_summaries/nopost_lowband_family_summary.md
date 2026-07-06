# NoPost Feature Lowband Family Summary

Date: 2026-07-06

Status: WLDB-A, the tested v2.18/v2.19/v2.20 deployable lowband predictor
forms, the v2.23 small-adapter train form, the v2.25A direct risk soft-label /
scale-distillation form, the v2.26 current-risk-input trainability route, the
v2.27 same-sample ILFRB-ACS action-bank construction, the first v2.28 OOF
prototype bank, the v2.29 safe-envelope / GT-free table-policy bank, the v2.30
compatibility-gated LCB table-policy bank, the v2.31 target-only action-value
identifiability route, the v2.32 S5-only bounded internal low-frequency
correction-field canary, the v2.33 masked WDMamba-alpha0.5 S5-BILFCF
compression canary, the v2.34 direct WDMamba-on-crop teacher-delta projection
canaries, the v2.35 256 crop-input/full-image-slice target contract, and the
v2.36 alpha0.5 full600 same-context WDMamba substrate, and the v2.37 oracle
mask substrate as a bridge-training authorization are not training-authorized.
NoPost lowband remains directionally open, but v2.37 shows that even after a
tail-safe oracle mask exists, target-only no-op/unsafe separability is not
strong enough to launch P5, bridge training, canary80, or locked test.

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
  - `../experiment_cards/2026-07-05-haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy.md`
  - `../experiment_cards/2026-07-05-haze4k-v2-31-nopost-action-value-identifiability-audit.md`
  - `../experiment_cards/2026-07-05-haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field.md`
  - `../experiment_cards/2026-07-05-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit.md`
  - `../experiment_cards/2026-07-06-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit.md`
  - `../experiment_cards/2026-07-06-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit.md`
  - `../experiment_cards/2026-07-06-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability.md`
  - `../experiment_cards/2026-07-06-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation.md`
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
  - `../experiment_logs/haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705/`
  - `../experiment_logs/haze4k_v2_31_nopost_action_value_identifiability_audit_20260705/`
  - `../experiment_logs/haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705/`
  - `../experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705/`
  - `../experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/`
  - `../experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/`
  - `../experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/`
  - `../experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/`

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
| Haze4K v2.30 NoPost ILFRB-ACS Compatibility-Gated OOF Table Policy | P0 passed architecture-delta audit with no model-structure change from v2.29. P2A removed accepted hard-to-easy and cross-bucket risk in the strict diagnostic set (`0.0/0.0`) and kept a useful safe-set restricted oracle: mean/hard/easy `+0.6749/+1.2371/+0.1552`, p05 `0.0`, CVaR5 `0.0`, severe `0`. The fold-out GT-free LCB table policy was not deployable: mean/hard/easy `+0.0141/+0.0565/+0.0000`, p05 `-0.0503`, CVaR5 `-0.3056`, severe `0.0375`, fold-tail pass `4/5`, table pass `0`, safe-set-to-table mean gap `+0.6608`. P2B was skipped, training was not launched, and locked test was untouched. | `P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE`; do not probe selector, train, or touch locked test. Follow-up must improve GT-free compatibility features/ranking/no-op thresholding before selector work. |
| Haze4K v2.31 NoPost Target-Only Action-Value Identifiability Audit | P0 passed architecture-delta audit with no model-structure change from v2.30. P2A added target-only physics/frequency/internal/A0-diagnostic features, nested fold-out ranking, physics-cluster bank diagnostics, no-op risk-coverage, leakage controls, and action-confusion evidence. The feature gate failed: combined useful_gt_0p30 all/hard AUROC `0.6854/0.5444`, easy should-noop AUROC `0.5934`, and fold std `0.1059`. The best nested ranker improved over the v2.30 table with mean/hard/easy `+0.4234/+0.8152/+0.0995`, but p05/CVaR5/severe `-0.4421/-1.2582/0.1375` failed tail safety. Safe-set oracle remained higher at `+0.6749/+1.2371/+0.1553`, so action value exists but is not deployably identifiable enough. P2B was skipped, training was not launched, and locked test was untouched. | `P2A_FAIL_ACTION_VALUE_IDENTIFIABILITY_CLOSE_CURRENT_BANK`; close the current discrete action-bank selector route and pivot to a separate bounded internal low-frequency correction-field design or materially new action-value signal. |
| Haze4K v2.32 NoPost Bounded Internal Low-Frequency Correction Field | P0 passed the new architecture contract from official anchor: strict partial-load loaded `602` official keys, only `8` `BILFCF_` keys were newly initialized, identity max/mean diff were `0.0`, forbidden symbol hits were `0`, and runtime remained `forward(self, x)`. P1 passed bounded-field sanity with tiny low-frequency activity: field mean/p95 `5.6611e-06/1.1682e-05`, high-frequency leakage `0.02363`, and gate mean `0.01822`. P2 canary32 then failed after `40` train-derived adapter-only steps: mean/hard/easy deltas `-0.4146/-0.3287/-0.4371`, p05/CVaR5/severe `-1.7719/-2.0842/0.5000`. P2 canary80 OOF and P3 objective ablation were not launched. | `P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`; do not expand this S5-only alpha=0.02 loss_C adapter-only BILFCF route by more epochs/folds/simple tuning. A follow-up must materially change the bounded-field training design before reopening. |
| Haze4K v2.33 NoPost Teacher-Benefit Source and BILFCF Trainability Audit | P1 found table-supported teacher-source benefit for `wdmamba_alpha0p5` and `wdmamba_full`. P2 crop-aligned sanity passed but showed only tiny local movement: GT one-image `+0.0124`, positive low-frequency delta `+0.0061`, sign-flip `-0.0119`. P3 showed S5 was not the largest amplification point, with `S6_decoder_early=0.0617`, `S5=0.0750`, and unsafe-high `decoder_pre_output_feature=0.5148`. P4 then tested WDMamba-alpha0.5 masked+preservation canary32 and failed the gate: mean/hard/easy `+0.0007/+0.0013/+0.0003`, p05/CVaR5 `-0.0025/-0.0029`, severe `0`, strong-reference regression `0`, eligible coverage `5/32`, and mask effect vs unmasked was negative. | `P4_FAIL_MASKED_CANARY32_NO_CANARY80`; teacher-source evidence remains useful, but this S5-BILFCF compression setup did not convert it into measurable micro-canary utility. Do not launch canary80 or locked test. |
| Haze4K v2.34 NoPost Teacher-Delta Projection and Multi-Stage Bridge Audit | P0/P0B confirmed the direct WDMamba-on-256-crop canaries are not valid projection substrates: P0 first32 direct WDMamba-alpha0.5 mean/hard/easy `-2.3193/-1.5978/-2.1668`, P1 table join missing `25/32`; P0B table-positive balanced canary still had mean/easy `-2.4753/-4.0017` with p05/CVaR5 `-8.7433/-11.4269`. P0C then showed the old full-image teacher evidence remains valid: table/full-image recompute alpha0.375/0.5 means `+4.1392/+5.7567`, full-image-output crop slices `+3.9106/+5.2963`, but direct crop inference `-1.4741/-2.4753` with negative direct-vs-fullslice context gap for `32/32`. | `P0B_FAIL_BALANCED_CANARY_DIRECT_TEACHER_GATE`; do not run free-tensor projection, generator gap, gradient conflict, bridge micro-canary, canary80, or locked test from direct-crop WDMamba canaries. Full-image WDMamba tables remain valid for their own view but are unreliable selectors for direct crop inference. |
| Haze4K v2.35 Full-Image Teacher Cache and Context-Contract Audit | P0D closed the 256 crop-input/full-image-slice target contract after rebasing against crop-direct A0: alpha0.5 mean/p05/CVaR5/severe_rate `-1.7067/-6.7084/-7.4537/0.625`. P1 passed the 600-image full-image cache/hash audit with `1200` alpha rows and table-vs-recompute mean/max abs diff `0.0/0.0`. P2 found valid same-context contracts: 384 alpha0.5 mean/p05/CVaR5 `+3.5217/+0.5167/+0.4038`, and best full-image-slice alpha0.5 `+5.2963/+2.0773/+1.0368`. P3 passed same-contract substrate construction with `32/32` positive samples. P4 passed all tested same-contract free-tensor projection insertion groups after archiving two engineering-invalid NaN attempts; best `S4_plus_S6` projection_ratio_vs_teacher `1.0090`, free mean `+5.3438`, p05 `+2.1914`, severe `0`. | `P4_PASS_SAME_CONTRACT_FREE_TENSOR_PROJECTION`; full-image/full-image-slice WDMamba is a valid same-context teacher substrate for a future written generator/bridge route. Do not train a 256 crop-input student on full-image-slice targets, and do not launch bridge training, canary80, or locked test from v2.35 without a new same-context route. |
| Haze4K v2.36 Same-Contract WLFBridge-S4S6 Generator Trainability Audit | P0 expanded the v2.35 alpha0.5 full-image same-context teacher distribution to `600` train-derived images. Mean/hard/easy remained strongly positive (`+3.2299/+4.9092/+1.1266 dB`) and cache sha coverage was `1.0`, but tail safety failed: p05/CVaR5 `+0.0084/-0.7438 dB`, severe_rate `0.035`, strong_reference_regression_rate `0.1733`, fold_pass `0/5`. Post-run audit recomputed `30` negative deltas, `21` severe regressions, and `26/150` strong-reference regressions. | `P0_FAIL_STOP_BEFORE_BRIDGE_TRAINING`; do not launch P0B context384 projection, P1 architecture identity, P2 generator fit, P3 OOF, canary80, or locked test from the current alpha0.5 full600 substrate. |
| Haze4K v2.37 Tail-Safe Same-Context WDMamba Eligibility and Preservation Audit | P0 found no unmasked alpha passed full600 safety. Alpha0.125 removed severe regressions but still had `3` strong-reference regressions and fold pass `2/5`; alpha0.5 reproduced the v2.36 tail failure. P1 confirmed `28/30` alpha0.5 negatives were easy or strong-reference. P2 passed an oracle teacher-positive + A0-preservation mask (`M0_oracle_positive`) with mean/hard/easy `+3.2671/+4.9091/+1.2569`, p05/CVaR5 `+0.0106/0.0`, eligible `570/600`, negative/severe preservation `1.0`, fold pass `5/5`. P3 OOF mask selection passed `5/5`, selecting M0 on every fold. P4 target-only no-op/unsafe separability failed: AUROC `0.8683`, AUPRC `0.2179`, severe recall at FPR0.10 `0.5714`, strong-reference unsafe recall `0.5769`, easy no-op precision `0.2857`, fold pass `0/5`. | `P4_FAIL_STOP_TARGET_ONLY_NOOP_UNSAFE_NOT_SEPARABLE`; the oracle mask substrate is tail-safe, but not deployably identifiable. Do not launch P5 masked free-tensor projection, bridge/generator training, canary80, or locked test. |

## Family Verdict

The precise conclusion is: close WLDB-A, v2.18 WLDB-A2 global pooled policy,
v2.19 O2 final-only spatial predictor, v2.20 O3 mid+final/global-context
predictor, v2.23 small-adapter train form, v2.25A direct risk soft-label /
scale-distillation, v2.26 current-risk-input calibration/trainability, and the
current v2.27 oracle-derived action-bank construction, the first v2.28 OOF
prototype bank, the v2.29 safe-envelope / GT-free table-policy bank, the v2.30
compatibility-gated LCB table-policy bank, the v2.31 target-only action-value
identifiability route, the current v2.32 S5-only BILFCF canary, the v2.33 /
v2.34 direct-crop WDMamba-alpha teacher canary line, the v2.35 256
crop-input/full-image-slice target contract, the v2.36 alpha0.5 full600
same-context WDMamba substrate, and the v2.37 oracle mask substrate as bridge
training candidates. Do not close
the underlying lowband-capacity direction, but close the current discrete
action-bank selector line unless a materially new target-only action-value
signal appears. The first bounded internal low-frequency correction-field
screen escaped the selector bottleneck mechanically, but failed
trainability/tail gates in canary32. The WDMamba teacher-source table remains
useful as route context, P0C showed the full-image table/recompute view remains
positive, and v2.35 showed full-image/full-image-slice WDMamba is representable
by same-context free tensors. v2.36 then showed the current alpha0.5 full600
substrate is not tail-safe enough for bridge training. v2.37 fixed that part at
the oracle mask level: M0 teacher-positive + A0-preservation passed P2 and P3
fold-stable substrate gates. The remaining blocker is deployability:
target-only no-op/unsafe separability failed P4, so no P5, bridge/generator,
canary80, or locked-test work is authorized. A follow-up must first introduce a
materially better target-only no-op/unsafe signal or a more conservative
hard-only contract before any generator/bridge training.

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

v2.30 tested the more precise post-v2.29 question: whether compatibility gates,
hard-to-easy firewalls, local strength dose response, and an LCB-risk table can
turn the OOF bank into a deployable GT-free policy. It improved the safety
interpretation: strict accepted hard-to-easy and cross-bucket diagnostic risk
were both `0.0`, and the safe-set restricted oracle stayed meaningful,
especially for hard samples. The deployable table still failed because it could
not rank/select the useful safe-set actions: mean and hard gain were near zero,
severe rate slightly exceeded the cap, and the safe-set-to-table mean gap was
`+0.6608`. That keeps the bank scientifically interesting but blocks P2B,
selector training, model training, and locked-test use.

v2.31 directly tested whether richer target-only physics/frequency/internal
features could rescue that action-value ranking bottleneck. They improved the
best nested ranker over the v2.30 table and recovered some hard gain, but failed
the written useful-action, easy-noop, fold-stability, and tail-safety gates.
The evidence therefore changes the family verdict from "improve GT-free table
selection before selector work" to "close the current discrete action-bank
selector route unless a materially new action-value signal is introduced."

v2.32 tested the correct post-v2.31 pivot: a zero-init, bounded-amplitude,
spatial internal low-frequency correction field inside ConvIR-B, with no action
bank, no ranker, no selector, and no locked-test feedback. P0/P1 were clean:
official weights partial-loaded exactly except for `BILFCF_`, identity was
exact, and the field stayed tiny and low-frequency after warmup. The first
adapter-only canary32 still failed because utility and tail moved the wrong
way. This is a different failure from v2.31: not action-value identifiability,
but trainability/safety failure for the current S5-only bounded-field design.

v2.33 separated the post-v2.32 questions. The WDMamba teacher-source audit
passed, and P2/P3 ruled out the simplest sign/scale and catastrophic S5
amplification explanations. The decisive P4 micro canary still failed because
masked teacher-benefit distillation into the current S5-BILFCF carrier produced
near-zero utility and no positive mask effect. This blocks canary80 and keeps
locked test untouched.

v2.34 tested the required P4 canary coverage/open-question before launching a
free-tensor upper bound. The direct-crop answer was negative: the exact v2.33
first32 canary had direct WDMamba-alpha0.5 crop benefit below zero, and a
rebuilt table-positive balanced canary still had strongly negative mean/easy
direct benefit. P0C then separated the cause: full-image WDMamba table/recompute
and full-image-output crop slices were strongly positive, while rerunning
WDMamba directly on 256 crops caused the failure. This means the next valid
projection route must first define a same-context teacher-positive diagnostic
canary, not reuse full-image table labels for direct crop inference or skip
ahead to P1/P2/P3/P4.

v2.35 ran that same-context audit instead of continuing the failed v2.34
direct-crop route. It closed Contract C for a 256 crop-input student trained
against full-image-output slices: after rebasing against crop-direct A0, the
alpha0.5 target was strongly negative with severe_rate `0.625`. It then passed
the full-image teacher cache/hash audit, found valid same-context contracts at
384 context and full-image-slice context, verified a `32/32` positive
same-contract substrate, and passed a free-tensor projection upper bound across
all tested insertion groups. This changes the route memory: WDMamba-alpha0.5 is
not "invalid" as a teacher, but it is only valid under a matched
full-image/full-image-slice context contract. Future bridge/generator work must
be written as a new same-contract route and cannot inherit authorization from
the failed direct-crop or 256 crop-input contracts.

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
- Do not continue the current v2.30 bank by probing a selector, training, or
  relaxing gates; the compatibility gate helped safety diagnostics, but the
  deployable GT-free LCB table policy still fails utility/severe gates.
- Do not continue the current v2.31 bank by adding more handcrafted features,
  ranker variants, table thresholds, selector probes, training, or locked-test
  checks; the target-only action-value identifiability and tail-safety gates
  failed.
- Do not continue the current v2.32 S5-only alpha=0.02 loss_C adapter-only
  BILFCF canary by launching canary80/P3 anyway, adding epochs/folds, relaxing
  tail gates, or using locked-test feedback; canary32 failed normally.
- Do not continue the v2.33 masked WDMamba-alpha0.5 S5-BILFCF compression
  route by launching canary80, increasing micro-canary steps/samples, or
  simple loss/mask tuning; P4 converted teacher benefit into near-zero utility.
- Do not continue v2.34 by launching free-tensor projection, generator gap,
  gradient conflict, bridge micro-canary, canary80, or locked test from the
  current first32 or table-selected balanced direct-crop WDMamba canaries; both
  failed the direct teacher-benefit gate, and P0C showed the failure is a
  crop-inference context mismatch.
- Do not continue v2.35 by training a 256 crop-input student on
  full-image-slice targets, treating same-contract free-tensor projection as a
  deployable bridge, or launching bridge training, generator training, canary80,
  or locked test without a new written same-context route.
- Do not treat the v2.21 replay pass, v2.25A AP/ECE partial passes, or the
  invalid tuple-sort AP as train-time success.
- Do not treat mean or hard-bucket improvement as sufficient if p05, CVaR,
  severe count, strong-reference regressions, or easy preservation fail.
- Do not use locked Haze4K feedback to tune lowband actions, objectives,
  thresholds, checkpoints, or route choice.
- Do not reopen the older NoPost severe-risk selector line unless it introduces
  a materially new signal beyond the v2.13-v2.15 failures.

## Reopen Condition

A credible follow-up must either introduce a materially new target-only
action-value signal before reopening a discrete action-bank selector, or
materially change the bounded-field training design beyond the current v2.32
S5-only alpha=0.02 loss_C adapter-only canary and the v2.33/v2.34
WDMamba-alpha0.5 crop-aligned teacher-canary failures. It must not be a WLDB-A
rerun, a direct v2.19/v2.20 training launch, a v2.23 expansion, a v2.25A
hyperparameter extension, a v2.26 rescue/action-joint continuation, a v2.27
selector launch from the same-sample bank, a v2.28 selector launch from the
current unsafe OOF prototype bank, a v2.29 selector launch from the failed
safe-envelope/table-policy bank, a v2.30 selector launch from the failed
compatibility-gated LCB table policy, v2.31 table/ranker micro-tuning, a simple
v2.32 rerun with more epochs/folds/loss-weight tuning, a v2.33 masked
teacher-compression rerun with more steps/samples/simple mask tuning, or a
v2.34 projection launch from a direct-crop WDMamba canary whose direct teacher
benefit gate failed, a v2.35 256 crop-input/full-image-slice target training
launch whose rebased contract failed, or a v2.36 bridge/generator launch from
the alpha0.5 full600 substrate whose CVaR/severe/strong-reference P0 gate
failed. A WDMamba follow-up may use the v2.35/v2.36 evidence only by writing a
new same-context route that first defines and passes a full600 tail-safe
eligibility, no-op, or preservation contract. It should keep the source-clean
and identity contract requirements, avoid locked-test feedback for design
selection, and make train-derived tail safety plus the route-specific mechanism
gate the first hard gate before selector, training, or locked-test work.
