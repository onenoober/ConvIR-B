# Haze4K DTA-v3.7 U-TQS-Mix Evidence

Date: 2026-06-13

Status: `D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING`

Route card: `experience_docx/experiment_cards/2026-06-13-haze4k-dta-v3-7-u-tqs-mix.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Selected host: `convir-4090`.
- Host decision: `convir-4090` reachable and mostly idle; `convir-5090` SSH failed with `Permission denied (publickey,password)`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Source OOF table: `experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/v36_formal_oof_per_image_action_table.csv`.
- Source selector errors: `experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/formal_hrcs/v36_selector_error_table.csv`.
- Locked test: touched exactly once in D9 after D8 strict pass; the sealed policy failed promotion and no post-test tuning is allowed.

## Route Decision

DTA-v3.7 is the new DTA mainline. The route abandons hard reject as the main
strategy and moves to an A0-preserving utility-aware soft action mixture with
transmission, airlight, quality, and uncertainty signals.

## Required Phase A Artifacts

- `status.txt`
- `run_dta_v3_7_phase_a_convir4090.sh`
- `v37_phase_a.log`
- `v37_positive_loss_budget_report.csv`
- `v37_soft_action_bank_oracle_grid.csv`
- `v37_false_reject_false_accept_taxonomy.csv`
- `v37_feature_ablation_auc_report.csv`
- `v37_tA_quality_uncertainty_preflight.json`
- `v37_phase_a_summary.json`

## Phase A Gate

Pass requires at least one soft action oracle row with coverage `>=0.95` that
passes mean, hard, dSSIM, positive-ratio, true-vs-controls, worst, and max outer
worst strict gates. The Phase A soft-alpha metrics are table-only linear-delta
proxies; real blended-output verification is required in Phase C before any
promotion claim.

## Phase A Completion

Phase A completed on `convir-4090` at `2026-06-13T11:44:49+08:00` with marker:

```text
DTA_V3_7_U_TQS_MIX_PHASE_A_OK rows=27000 soft_rows=18 strict_soft=13 gate=PASS_SOFT_ORACLE_HEADROOM
```

Primary row:

| Bank | Utility | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | intervention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_L2_L3_L1_full` | `max_dpsnr` | `+0.143298` | `+0.121101` | `+0.00002551` | `0.6623` | `0.00` | `0.00` | `0.6623` |

Budget and preflight findings:

- L3 hard-reject positive loss is `32.73/600`, about `8.77x` the strict budget.
- L1 hard-reject positive loss is `29.87/600`, about `6.79x` the strict budget.
- Transmission GT/pred/uncertainty columns are present; explicit airlight GT and NR-IQA features are not yet present.
- Current deployable severe-risk AUC remains weak at about `0.608`, so Phase B must add T/A/Q/U feature separability.

Decision: `PHASE_A_PASS_SOFT_ORACLE_HEADROOM`; proceed to Phase B.


## Phase B TQS Plan

Run script: `run_dta_v3_7_phase_b_tqs_convir4090.sh`.

Outputs:

- `status_phase_b_tqs.txt`
- `v37_phase_b_tqs.log`
- `v37_tqs_policy_nested_report.csv`
- `v37_tqs_policy_aggregate.csv`
- `v37_tqs_policy_action_table.csv`
- `v37_tqs_feature_group_ablation.csv`
- `v37_tqs_summary.json`

This is a train-derived table-only policy diagnostic. Deployable feature groups
must not use `trans_gt`; diagnostic `trans_gt` rows are reported only to measure
how much separability physical supervision could add.

## Phase B TQS Result

Phase B completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phaseb` with
marker:

```text
DTA_V3_7_TQS_PHASE_B_OK rows=27000 groups=6 strict_pass=0 decision=PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND
```

Best aggregate deployable row:

| Feature group | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | intervention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T_pred` | `+0.015792` | `+0.013137` | `-0.00000566` | `0.6360` | `0.80` | `2.33` | `0.9993` |

Interpretation:

- The table-only predictor can control severe tail, but it collapses gain: mean,
  hard, dSSIM, and true-vs-controls fail strict gates.
- `T_pred` is the best current deployable group, which supports the T/A/Q/U
  direction, but existing table features are not enough to recover the Phase A
  oracle headroom.
- `diagnostic_with_trans_gt` also fails, so direct use of the existing
  transmission GT columns alone is not sufficient; real feature enrichment and
  blended-output verification are required.

Decision: `PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND`.
Proceed to feature enrichment / real soft-blend verification before formal
policy claims. Do not return to v3.6 hard-reject threshold search.


## Phase B2 Enriched TQS Plan

Added scripts:

```text
experience_docx/tools/extract_haze4k_v37_quality_features.py
experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/run_dta_v3_7_phase_b2_enriched_tqs_convir4090.sh
```

This phase extracts deployable image quality, contrast, dark-channel, edge,
texture, sky/highlight, and color-cast features from Haze4K train-derived hazy
images, joins them to the OOF action table, and reruns nested TQS with
`v37_tqs_enriched_*` outputs. It does not use locked test feedback.

## Phase B2 Enriched TQS Result

Phase B2 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phaseb2` with
marker:

```text
DTA_V3_7_QUALITY_FEATURES_OK images=3000 missing=0
DTA_V3_7_TQS_PHASE_B_OK rows=27000 groups=7 strict_pass=0 decision=PHASE_B_TABLE_POLICY_STRICT_FAIL_NEEDS_FEATURE_ENRICHMENT_OR_REAL_BLEND
```

Key rows:

| Feature group | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | intervention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T_pred` | `+0.015792` | `+0.013137` | `-0.00000566` | `0.6360` | `0.80` | `2.33` | `0.9993` |
| `deployable_TQAU_action_all` | `+0.021754` | `+0.024839` | `+0.00000301` | `0.5128` | `3.07` | `4.33` | `0.7718` |
| `diagnostic_with_trans_gt` | `+0.022158` | `+0.025835` | `+0.00000264` | `0.5006` | `3.13` | `4.67` | `0.7457` |

Interpretation:

- The new image quality/color/edge/dark-channel features were extracted for all
  `3000` train hazy images with zero missing files.
- Enriched deployable features improve the gain/dSSIM tradeoff relative to the
  first table-only policy, but still fail strict mean, hard, positive-ratio, and
  true-vs-control gates.
- The bottleneck is no longer severe-tail control; tail is easy. The bottleneck
  is preserving enough positive/high-gain action while staying deployable.

Decision: `PHASE_B2_ENRICHED_TABLE_POLICY_STRICT_FAIL`. Continue to real
soft-blend verification and/or integrated T/A/U supervised candidate training;
do not return to hard-reject threshold search.


## Phase C1 Real Soft-Blend Verification Plan

Added scripts:

```text
experience_docx/tools/eval_haze4k_dta_v37_real_blend_oracle.py
experience_docx/tools/aggregate_haze4k_dta_v37_real_blend_oracle.py
experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/run_dta_v3_7_phase_c1_real_blend_convir4090.sh
```

This phase replaces Phase A's linear metric proxy with actual rendered tensor
blends:

```text
blend = clamp(A0 + alpha * (candidate - A0), 0, 1)
```

Scope:

- train-root Haze4K OOF fold validation images only; locked test remains untouched.
- variants: L2 tail-safe, L3 balanced, L1 high-gain from the v3.6 formal checkpoint family.
- groups: folds `0..4` x seeds `3407,3411,2026`.
- bank specs: A0/L3 full, A0/L3 shrink, A0/L2/L3/L1 full, A0/L2/L3/L1 shrink, micro-shrink, and forced no-A0 shrink.
- controls: same selected action evaluated against zero, shuffled, and normal depth modes for true-vs-control surplus.

Required outputs:

```text
status_phase_c1_real_blend.txt
v37_phase_c1_real_blend_*.log
phase_c1_real_blend_groups/v37_real_blend_selected_seed*_f*.csv
phase_c1_real_blend_groups/v37_real_blend_single_actions_seed*_f*.csv
phase_c1_real_blend_groups/v37_real_blend_summary_seed*_f*.json
v37_real_blend_oracle_selected_all.csv
v37_real_blend_oracle_grid.csv
v37_real_blend_summary.json
```

Pass line is the same strict oracle line as Phase A, but now with actual
PSNR/SSIM computed from blended image tensors and with `max_outer_worst`
measured across fold-seed 600-image groups.


## Phase C1 Real Soft-Blend Verification Result

Phase C1 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phasec1` with marker:

```text
DTA_V3_7_PHASE_C1_REAL_BLEND_OK 2026-06-13T12:47:32+08:00
```

Aggregate result:

```text
DTA_V3_7_REAL_BLEND_AGGREGATE_OK rows=162000 grid=18 strict_pass=14 decision=PHASE_C1_REAL_BLEND_ORACLE_PASS
```

Best actual blended-image oracle row:

| Bank | Utility | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | max outer worst/600 | true-vs-zero | true-vs-shuffle | true-vs-normal | intervention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_L2_L3_L1_micro_shrink` | `max_dpsnr` | `+0.143568` | `+0.121118` | `+0.00002579` | `0.6977` | `0.00` | `0.00` | `+0.106861` | `+0.080749` | `+0.088555` | `0.6977` |

Key follow-up rows:

| Bank | Utility | strict | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | intervention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_L2_L3_L1_shrink` | `max_dpsnr` | `true` | `+0.143564` | `+0.121118` | `+0.00002579` | `0.6919` | `0.00` | `0.6919` |
| `A0_L2_L3_L1_full` | `max_dpsnr` | `true` | `+0.143298` | `+0.121101` | `+0.00002551` | `0.6623` | `0.00` | `0.6623` |

Interpretation:

- Phase A was not an over-optimistic metric artifact: actual tensor blending preserves and slightly improves the oracle headroom.
- Shrink/micro-shrink raises positive ratio from `0.6623` to `0.6977` with zero severe regressions under the oracle, directly validating the strategy-space change away from hard reject.
- The remaining bottleneck is deployable selection/mixing, not candidate-family headroom or image-space blending.
- Large raw per-image selected/action CSVs were kept on `convir-4090` under `phase_c1_real_blend_groups/` and `v37_real_blend_oracle_selected_all.csv`; GitHub sync keeps compact aggregate grid, summaries, status, and logs to avoid committing oversized raw metric tables.

Decision: `PHASE_C1_REAL_BLEND_ORACLE_PASS`. Proceed aggressively to integrated
T/A/U supervised candidate training plus deployable U-TQS soft-mix policy; do
not return to v3.6 hard-reject threshold tuning.


## Phase D1 Integrated T/A/U Candidate Training Plan

Added code and scripts:

```text
Dehazing/ITS/models/ConvIR.py
Dehazing/ITS/train.py
Dehazing/ITS/main.py
experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/run_dta_v3_7_tau_candidate_convir4090.sh
experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/launch_dta_v3_7_tau_training_convir4090.sh
```

Model change:

- add `DTA.airlight_head` and `DTA.airlight_uncertainty_head`;
- supervise atmospheric light from Haze4K filename metadata with
  `--dta_airlight_weight` and `--dta_airlight_nll_weight`;
- keep transmission supervision through `--dta_trans_log_weight` and
  `--dta_trans_nll_weight`;
- keep quality/risk pressure through light-tail, CVaR, group-tail, and patch-SSIM
  losses;
- keep A0 preservation through official A0 partial initialization,
  reference-preserve, MSE-regression, and bounded FDF action budgets.

Partial-load and initialization rule:

```text
init_model = official A0 Haze4K checkpoint
init_model_partial = true
partial_new_prefixes = DTA.
new modules = all DTA modules, including airlight/airlight-uncertainty heads
airlight final layers = zero initialized, sigmoid output starts at 0.5
output path = unchanged unless trained DTA/FDF action changes it
```

Queue:

```text
variants = u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006
folds = 0,1,2,3,4
seeds = 3407,3411,2026
stage = quick5full
locked_test_touched = false
```

This is deliberately not a conservative detour: it is the direct integrated
T/A/U candidate retraining authorized by Phase C1.

## Phase D1 Integrated T/A/U Staged Screen Result

D1 staged-screen evidence completed on `convir-4090` with locked test untouched.
The completed screen is `3` variants x folds `0,1` x seeds `3407/3411` = `12`
train-derived quick5full candidates. The original broad queue was stopped as a
protocol correction; the missing/incomplete jobs were repaired with `screen3`,
`screen4`, and a direct u1 fold0 eval repair. The final compact summaries are:

```text
v37_tau_stage_screen_decision.txt
v37_tau_stage_screen_matrix_summary.csv
v37_tau_stage_screen_matrix_summary.json
v37_tau_stage_screen_run_matrix_rows.csv
v37_tau_oof_per_image_action_table.csv
v37_tau_oracle_risk_coverage_curve.csv
v37_tau_selector_nested_calibration_report.csv
```

Stage-screen aggregate over the intended 12 runs:

| variant | runs | mean | hard | dSSIM | positive | worst/600 | true-vs-zero | screen strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `u1_tau_l1_s004_g025_a006` | 4 | `+0.069596` | `+0.076433` | `+0.00000786` | `0.6412` | `68.75` | `+0.089704` | fail tail |
| `u2_tau_l3_s004_g015_a006` | 4 | `+0.059690` | `+0.067173` | `+0.00000992` | `0.6408` | `57.25` | `+0.073943` | fail tail |
| `u3_tau_l2_s002_g025_a006` | 4 | `+0.048004` | `+0.054895` | `+0.00000988` | `0.6346` | `49.00` | `+0.059999` | fail mean/tail |

Interpretation:

- Integrated T/A/U supervision is mechanism-positive: all staged variants keep positive mean, hard, dSSIM, positive ratio, and true-vs-zero surplus.
- No D1 variant is promotion-ready: `u1/u2` recover gain but fail the severe-tail gate, while `u3` is closest on tail but misses the mean gate and still has `49/600` worst.
- Do not run full `5 folds x 3 seeds` for these raw D1 candidates yet, and do not touch locked Haze4K test.
- Next work should use the D1 evidence as candidate/feature material for deployable U-TQS soft-mix/shrink policy and tail-aware action mixing, not resume hard reject or broad router-capacity search.

Decision: `D1_STAGE_SCREEN_TRIAGE_COMPLETE_NO_FORMAL_PROMOTION_YET_LOCKED_TEST_UNTOUCHED`.

## Phase D2 TAU Soft-Shrink Policy Plan

D2 is the next staged step after D1. It does not promote any raw D1 candidate to
full formal. Instead, it uses the completed D1 table as candidate/feature
material for an A0-preserving U-TQS soft-shrink policy:

```text
input = v37_tau_oof_per_image_action_table.csv
scope = D1 intended screen only
variants = u1/u2/u3
folds = 0,1
seeds = 3407,3411
run filter = quick5full only
action_bank = A0 + alpha * {u1,u2,u3}, alpha in {0.10,0.25,0.50,0.75,1.00}
locked_test_touched = false
```

Artifacts:

```text
experience_docx/tools/train_haze4k_dta_v37_tau_shrink_policy.py
run_dta_v3_7_phase_d2_tau_shrink_policy_convir4090.sh
v37_tau_shrink_oracle_grid.csv
v37_tau_shrink_policy_nested_report.csv
v37_tau_shrink_policy_aggregate.csv
v37_tau_shrink_policy_action_table.csv
v37_tau_shrink_summary.json
```

This is a required intermediate result, not a conservative detour. A D2 passing
table policy would still need real rendered soft-blend verification before any
formal claim, because D2 alpha-shrink uses table-scaled D1 deltas.

## Phase D2 TAU Soft-Shrink Policy Result

Phase D2 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d2-policy-c823848`
with locked test untouched:

```text
DTA_V3_7_D2_TAU_SHRINK_POLICY_OK rows=7200 oracle_strict=10 policy_strict=0 decision=D2_TAU_SHRINK_POLICY_STRICT_FAIL
```

The run explicitly filtered to `quick5full` D1 screen rows:

```text
raw_rows = 7802
filtered_rows_before_dedup = 7200
filtered_rows_after_dedup = 7200
include_run_substring = quick5full
```

Top table-oracle row:

| bank | utility | mean | hard | dSSIM | positive | worst/600 | max outer worst/600 | true-vs-zero |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | `max_dpsnr` | `+0.136286` | `+0.120366` | `+0.00002506` | `0.6742` | `0.00` | `0.00` | `+0.115843` |

Best deployable nested policy row:

| feature group | bank | mean | hard | dSSIM | positive | worst/600 | true-vs-zero | strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `T_pred` | `micro_shrink` | `+0.006518` | `+0.007339` | `+0.00000089` | `0.6400` | `0.00` | `+0.008293` | fail mean/hard/control surplus |

Useful follow-up deployable rows:

| feature group | mean | hard | positive | worst/600 | true-vs-zero | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `deployable_TQAU_action_all` | `+0.012957` | `+0.015138` | `0.5408` | `3.25` | `+0.014510` | fail gain/positive/control |
| `FDF_action_stats` | `+0.010768` | `+0.018220` | `0.5625` | `5.00` | `+0.014457` | fail gain/positive/control |

Interpretation:

- D2 confirms strong D1 action-family headroom under per-image oracle selection:
  `10/12` oracle rows strict-pass and best oracle has zero severe tail.
- D2 deployable table policy still strict-fails: severe-tail control is easy,
  but deployable features collapse mean/hard/control surplus by choosing too many
  tiny actions.
- Do not run raw D1 full `5x3`, and do not touch locked test.
- The next experiment should be a real rendered soft-blend verification for a
  fixed D1 action bank plus a stronger policy target that preserves high-gain
  positives instead of minimizing tail at the cost of almost all utility.

Decision: `D2_TAU_SHRINK_POLICY_STRICT_FAIL_ORACLE_PASS_LOCKED_TEST_UNTOUCHED`.

## Phase D3 TAU Real Soft-Blend Verification Plan

D3 is the immediate follow-up to D2. It does not run raw D1 candidates at full
`5 folds x 3 seeds` and does not touch locked Haze4K test. It renders actual
image-space blends for the D1 quick5full candidate bank so the D2 table-scaled
alpha assumption is checked with real tensors.

```text
scope = D1 quick5full screen only
variants = u1/u2/u3
folds = 0,1
seeds = 3407,3411
action bank = A0 + alpha * {u1,u2,u3}
alpha bank = 0.10,0.25,0.50,0.75,1.00
utility modes = max_dpsnr, tail_averse, ssim_guarded, high_positive_tail_averse
locked_test_touched = false
```

Added artifacts:

```text
experience_docx/tools/eval_haze4k_dta_v37_tau_real_blend_oracle.py
experience_docx/tools/aggregate_haze4k_dta_v37_tau_real_blend_oracle.py
run_dta_v3_7_phase_d3_tau_real_blend_convir4090.sh
```

Pass interpretation: if D3 actual rendered oracle passes strict gates, continue
with a stronger high-positive deployable utility policy over the same D1 action
bank. If D3 actual rendered oracle fails, the D2 table oracle was too optimistic
and the next step must redesign candidate/blend representation rather than run
formal `5x3` or locked test.


## Phase D3 TAU Real Soft-Blend Verification Result

D3 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d3-realblend-40b4db8`
with locked test untouched:

```text
DTA_V3_7_D3_TAU_REAL_BLEND_AGGREGATE_OK rows=57600 grid=24 strict_pass=22 decision=PHASE_D3_TAU_REAL_BLEND_ORACLE_PASS
DTA_V3_7_PHASE_D3_TAU_REAL_BLEND_OK
```

Top actual rendered D1 soft-blend oracle row:

| bank | utility | mean | hard | dSSIM | positive | worst/600 | max outer worst/600 | true-vs-zero | true-vs-shuffle | true-vs-normal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A0_U3_U2_U1_micro_shrink` | `max_dpsnr` | `+0.136562` | `+0.120391` | `+0.00002539` | `0.7071` | `0.00` | `0.00` | `+0.116236` | `+0.087609` | `+0.095873` |

Interpretation:

- D3 confirms the D2 D1-action oracle was not just a table-scaling artifact:
  actual rendered tensor blending strict-passes with `22/24` strict rows.
- Micro-shrink over `A0+u3/u2/u1` raises positive ratio to `0.7071` while
  keeping severe tail at `0/600`, so the route should keep pursuing soft-mix
  utility rather than raw D1 full `5x3` or hard reject.
- The remaining blocker is still deployable high-positive gain-risk prediction,
  not action-family headroom or severe-tail control.

Decision: `PHASE_D3_TAU_REAL_BLEND_ORACLE_PASS_LOCKED_TEST_UNTOUCHED`. Continue
immediately to a deployable high-positive utility policy over the D1/D3 action
bank; do not run raw D1 full `5x3` and do not touch locked test.

## Phase D4 High-Positive Deployable Policy Plan

D4 starts immediately after the D3 oracle pass. It is train-derived and table-only:
it uses actual D3 rendered single-action deltas plus deployable D1 T/A/U/FDF
features to train nested ridge/logistic high-positive utility policies.

```text
input actions = v37_tau_real_blend_single_actions_all.csv
input features = v37_tau_oof_per_image_action_table.csv
scope = D1/D3 quick5full only, folds 0,1, seeds 3407,3411
candidate banks = full, shrink, micro_shrink
feature groups = Q_input_proxy, T_pred, FDF_action_stats, deployable_TQAU_action_all, diagnostic_with_trans_gt
locked_test_touched = false
```

Artifacts:

```text
experience_docx/tools/train_haze4k_dta_v37_d3_high_positive_policy.py
run_dta_v3_7_phase_d4_highpos_policy_convir4090.sh
v37_d4_highpos_policy_selected_actions.csv
v37_d4_highpos_policy_aggregate.csv
v37_d4_highpos_policy_nested_report.csv
v37_d4_highpos_summary.json
```

D4 is not a formal claim. If a deployable row strict-passes, the next step is a
fixed staged confirmation before any full formal or locked-test discussion. If
it fails, the route remains blocked on deployable high-positive gain-risk
separability despite confirmed D3 oracle headroom.


## Phase D4 High-Positive Deployable Policy Result

D4 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d4-highpos-b7ab7c9`
with locked test untouched:

```text
DTA_V3_7_D4_HIGH_POSITIVE_POLICY_OK rows=38400 aggregate=75 strict_pass=0 decision=D4_HIGH_POSITIVE_POLICY_STRICT_FAIL
```

Best nested policy row:

| feature group | bank | policy | mean | hard | dSSIM | positive | worst/600 | true-vs-zero | intervention | strict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `diagnostic_with_trans_gt` | `micro_shrink` | `positive_breakthrough` | `+0.003344` | `+0.001875` | `+0.00000092` | `0.1588` | `0.00` | `+0.003334` | `0.2100` | fail gain/positive/control |

Interpretation:

- D4 confirms the naive deployable high-positive ridge policy is still far below
  the D3 oracle: it is tail-safe but collapses intervention and positive action
  rate.
- The issue is not severe-tail control; the issue is deployable ranking/forcing
  of high-positive actions.
- The next fast step should force target intervention/coverage bands on the D3
  action ranking to test whether the predictor has any useful ordering signal
  when it is not allowed to retreat to A0/tiny actions.

Decision: `D4_HIGH_POSITIVE_POLICY_STRICT_FAIL_LOCKED_TEST_UNTOUCHED`. Continue
with targeted-intervention D5 policy over the same D1/D3 action bank. Raw D1
full `5x3` and locked test remain blocked.

## Phase D5 Targeted-Intervention Policy Plan

D5 follows the D4 strict fail without pausing. It uses the same D1/D3 action bank
but forces target intervention bands (`0.35` to `1.00`) from deployable predicted
rankings, testing whether the D4 predictor has any useful ordering signal once it
is not allowed to retreat to A0/tiny-action behavior.

Artifacts:

```text
experience_docx/tools/train_haze4k_dta_v37_d5_targeted_intervention_policy.py
run_dta_v3_7_phase_d5_targeted_policy_convir4090.sh
v37_d5_targeted_policy_aggregate.csv
v37_d5_targeted_policy_nested_report.csv
v37_d5_targeted_summary.json
```

Locked Haze4K test remains untouched, and raw D1 full `5x3` remains blocked.


## Phase D5 Targeted-Intervention Policy Result

D5 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d5-targeted-b2a074e`
with locked test untouched:

```text
DTA_V3_7_D5_TARGETED_INTERVENTION_POLICY_OK rows=38400 aggregate=360 strict_pass=0 decision=D5_TARGETED_INTERVENTION_POLICY_STRICT_FAIL
```

Best targeted row:

| feature group | bank | score mode | target intervention | mean | hard | dSSIM | positive | worst/600 | true-vs-zero | true-vs-shuffle | true-vs-normal | strict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `diagnostic_with_trans_gt` | `shrink` | `pred_positive_forced` | `1.00` | `+0.019050` | `+0.016915` | `+0.00000532` | `0.6654` | `0.50` | `+0.020827` | `+0.015984` | `+0.016952` | fail gain/control |

Interpretation:

- Forcing intervention recovers positive ratio and keeps severe tail safe, but
  mean/hard/control surplus remain far below strict lines.
- Even the diagnostic feature group with transmission GT does not recover enough
  deployable ordering signal from the current D1 table features.
- The next route step should stop spending cycles on table-only threshold/forcing
  policies and add deployable actual candidate-vs-A0 output-difference / NR-IQA
  features, or train an integrated soft-mix head that sees candidate residual
  evidence directly.

Decision: `D5_TARGETED_INTERVENTION_POLICY_STRICT_FAIL_LOCKED_TEST_UNTOUCHED`.
Raw D1 full `5x3` and locked test remain blocked; continue with D6 output-diff /
quality feature extraction or an integrated soft-mix head rather than more hard
reject/table-only tuning.

## Phase D6 Output-Difference / Quality Policy Plan

D6 follows the D5 strict fail without pausing. It adds deployable actual
candidate-vs-A0 output-difference features from the rendered D1/D3 action bank,
then reruns nested targeted-intervention policies on the same train-derived
quick5full scope:

```text
scope = D1 quick5full candidates only
folds = 0,1
seeds = 3407,3411
locked_test_touched = false
raw D1 full 5x3 = blocked
```

Artifacts:

```text
experience_docx/tools/extract_haze4k_dta_v37_outputdiff_features.py
experience_docx/tools/train_haze4k_dta_v37_d6_outputdiff_policy.py
run_dta_v3_7_phase_d6_outputdiff_policy_convir4090.sh
status_phase_d6_outputdiff_policy.txt
v37_d6_outputdiff_summary.json
v37_d6_outputdiff_policy_aggregate.csv
v37_d6_outputdiff_policy_nested_report.csv
v37_d6_outputdiff_feature_groups.json
```

The combined raw feature table `v37_d6_outputdiff_features_all.csv` was generated
on `convir-4090` for reproducibility but is intentionally not synced as GitHub
evidence because it is a large raw runtime feature table (`~63.7 MB`).


## Phase D6 Output-Difference / Quality Policy Result

D6 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d6-outputdiff-91bcd32`
with locked test untouched:

```text
DTA_V3_7_D6_OUTPUTDIFF_POLICY_OK rows=38400 outputdiff_rows=36000 aggregate=522 strict_pass=8 decision=D6_OUTPUTDIFF_POLICY_STRICT_PASS
```

Best strict rows:

| feature group | bank | score mode | target intervention | mean | hard | dSSIM | positive | worst/600 | max outer worst/600 | true-vs-zero | true-vs-shuffle | true-vs-normal | strict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `outputdiff_plus_Q` | `micro_shrink` | `pred_gain` | `1.00` | `+0.078596` | `+0.085328` | `+0.00001913` | `0.6583` | `46.00` | `52.00` | `+0.083899` | `+0.064545` | `+0.068983` | pass |
| `outputdiff_only` | `micro_shrink` | `pred_gain` | `1.00` | `+0.078515` | `+0.086135` | `+0.00001889` | `0.6571` | `44.00` | `50.00` | `+0.082522` | `+0.063678` | `+0.068423` | pass |
| `deployable_TQAU_outputdiff_all` | `micro_shrink` | `pred_gain` | `1.00` | `+0.078253` | `+0.085323` | `+0.00001858` | `0.6558` | `44.25` | `50.00` | `+0.083757` | `+0.064409` | `+0.068986` | pass |

Interpretation:

- D6 is the first deployable v3.7 policy stage to strict-pass: `8/522` nested
  policy rows pass all strict gates on the D1 quick5full train-derived scope.
- The pass comes from actual candidate-vs-A0 output-difference features, proving
  that the D4/D5 failure was feature separability rather than lack of action
  headroom or a utility objective defect.
- The highest-mean strict row is `outputdiff_plus_Q / micro_shrink / pred_gain`
  with full intervention, mean `+0.078596`, hard `+0.085328`, positive ratio
  `0.6583`, worst `46/600`, and max outer worst `52/600`.
- `outputdiff_only` is nearly tied and slightly safer on the severe-tail count,
  so fixed-policy confirmation should compare the top deployable strict rows
  without reselecting from locked test feedback.

Decision: `D6_OUTPUTDIFF_POLICY_STRICT_PASS_LOCKED_TEST_UNTOUCHED`. Promote D6
to the next train-derived fixed-policy confirmation stage. Do not run locked
Haze4K test yet, and do not return to hard-reject threshold search. Raw D1 full
`5x3` remains blocked until a fixed D6 policy confirmation explicitly authorizes
the necessary additional candidate rendering.

## Phase D7 Fixed Output-Difference Policy Confirmation Result

D7 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d7-fixed-42228b6`
with locked test untouched and raw D1 full `5x3` still not run:

```text
DTA_V3_7_D7_FIXED_OUTPUTDIFF_CONFIRM_OK policies=2 strict_consistent=2 decision=D7_FIXED_OUTPUTDIFF_CONFIRM_PASS
```

Fixed policies checked:

| role | policy id | feature group | bank | score | target | mean | hard | dSSIM | positive | worst/600 | max outer worst/600 | strict | D6 consistency |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| primary | `primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` | `outputdiff_plus_Q` | `micro_shrink` | `pred_gain` | `1.00` | `+0.078596` | `+0.085328` | `+0.00001913` | `0.6583` | `46.00` | `52.00` | pass | pass |
| backup | `backup_outputdiff_only_micro_shrink_pred_gain_t100` | `outputdiff_only` | `micro_shrink` | `pred_gain` | `1.00` | `+0.078515` | `+0.086135` | `+0.00001889` | `0.6571` | `44.00` | `50.00` | pass | pass |

Interpretation:

- D7 freezes the D6 strict rows and reruns them as fixed policies; it does not
  perform another grid/policy search.
- Both fixed policies reproduce the D6 metrics exactly within tolerance and pass
  all strict gates on the D1 quick5full train-derived scope.
- The primary sealed candidate is
  `outputdiff_plus_Q / micro_shrink / pred_gain / target=1.00`; the
  `outputdiff_only` row remains a tail-safety backup.
- This finishes the current D1 quick5full staged policy-validation queue, but it
  is not yet a locked-test promotion: the policy still needs a broader
  train-derived formal confirmation before one locked-test confirmation can be
  justified.

Decision: `D7_FIXED_OUTPUTDIFF_CONFIRM_PASS_LOCKED_TEST_UNTOUCHED`. No active
D6/D7 training or audit remains. Next route step, if continuing, should be a
predeclared broader train-derived formal confirmation of the sealed D7 primary
policy; locked Haze4K test remains blocked until that formal confirmation passes.

## Phase D8 Fixed Formal Confirmation Outcome

D8 completed on `convir-4090` from runtime workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9`
with locked Haze4K test untouched. This stage did not introduce a new model
architecture and did not run policy search; it expanded only the sealed D7
primary policy to a broader predeclared train-derived confirmation scope.

Fixed D8 protocol:

```text
policy_id = primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100
policy_search = false
locked_test_touched = false
variants = u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006
folds = 0,1,2,3,4
seeds = 3407,3411,2026
stage = quick5full
run_tag = d8formal
```

Completion markers copied into local evidence:

```text
dta_v3_7_phase_d8_stage2_real_blend_done 2026-06-14T09:57:29+08:00
dta_v3_7_phase_d8_stage3_outputdiff_done 2026-06-14T10:23:58+08:00
dta_v3_7_phase_d8_stage2_fast_done rc=0 2026-06-14T10:24:18+08:00
DTA_V3_7_PHASE_D8_STAGE2_FAST_OK 2026-06-14T10:24:18+08:00
```

Final fixed-policy metrics recorded from the cloud closeout summary:

| policy id | coverage | mean | hard | easy | dSSIM | positive | worst/600 | max outer worst/600 | true vs zero | true vs shuffle | true vs normal | strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` | `1.0000` | `+0.078297` | `+0.085281` | `+0.062693` | `+0.00001904` | `0.65875` | `46.50` | `52.00` | `+0.083620` | `+0.064332` | `+0.068854` | pass |

Closeout counts:

```text
fixed_policy_count=1
strict_consistent_count=1
primary_pass=True
strict_gate_pass=True
locked_test_touched=False
raw_d1_full_5x3_run=True
image_groups=2400
rows=38400
outputdiff_rows=135000
```

Synced allowed text evidence includes the D8 launch manifest, status files,
launcher and stage logs, the poststage script, exact D8 final JSON/CSV artifacts,
and the local recovered closeout summary. Large raw output-difference feature tables, raw action tables,
checkpoints, candidate outputs, rendered images, and compare directories remain
cloud-only by default.

Cloud-generated exact final artifacts copied from `convir-4090` on 2026-06-14 12:02 CST:

- `v37_d8_fixed_formal_summary.json`
- `v37_d8_fixed_formal_policy_aggregate.csv`
- `v37_d8_fixed_formal_per_outer_report.csv`
- `v37_d8_fixed_formal_fixed_policy_config.json`
- `v37_d8_fixed_formal_selected_actions.csv`

Decision: `D8_FIXED_FORMAL_STRICT_PASS_LOCKED_TEST_UNTOUCHED`. The sealed policy
is now eligible for one fixed, one-shot locked Haze4K confirmation. The locked
result must not be used to tune thresholds, features, action membership,
checkpoints, or code.


## Phase D9 One-Shot Locked Fixed-Policy Confirmation Outcome

D9 completed on `convir-4090` from the D8 formal workspace
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-7-u-tqs-mix-d8-formal-5541ca9`.
This was the single pre-authorized locked Haze4K confirmation of the sealed D8
policy. It did not perform policy search and must not be followed by threshold,
feature, action-bank, checkpoint, or code tuning from the locked-test result.

Fixed locked protocol:

```text
policy_id = primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100
feature_group = outputdiff_plus_Q
action_bank = micro_shrink
score_mode = pred_gain
target_intervention = 1.00
outer_groups = 0:3407,0:3411,1:3407,1:3411
gpu_list = 1,2,3,4
locked_test_touched = true
one_shot_locked_confirmation = true
post_test_tuning_allowed = false
runner_source_commit = 8bf4030
```

Completion markers:

```text
DTA_V3_7_D9_GROUP_PARSE_OK groups=4 gpus=4 2026-06-14T12:28:06+08:00
d9_locked_group_done group=0:3407 rc=0 2026-06-14T12:48:33+08:00
d9_locked_group_done group=0:3411 rc=0 2026-06-14T12:48:35+08:00
d9_locked_group_done group=1:3407 rc=0 2026-06-14T12:48:35+08:00
d9_locked_group_done group=1:3411 rc=0 2026-06-14T12:48:35+08:00
DTA_V3_7_D9_LOCKED_FIXED_POLICY_OK decision=D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING rows=4000 images=1000 mean=0.020946 hard=0.021359 positive=0.531750 worst_per_600=35.70
DTA_V3_7_PHASE_D9_LOCKED_FIXED_POLICY_OK 2026-06-14T12:48:38+08:00
```

Final locked-test metrics:

| policy id | coverage | mean | hard | dSSIM | positive | worst/600 | true vs zero | true vs shuffle | true vs normal | intervention | strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` | `1.0000` | `+0.020946` | `+0.021359` | `+0.00004350` | `0.53175` | `35.70` | `+0.009704` | `+0.012502` | `+0.015170` | `1.0000` | fail |

Per locked outer group:

| fold | seed | mean | hard | dSSIM | positive | worst/600 | true vs zero | strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `3407` | `+0.026727` | `+0.026266` | `+0.00005852` | `0.5260` | `39.60` | `+0.012296` | fail |
| `0` | `3411` | `+0.025533` | `+0.025768` | `+0.00004718` | `0.5270` | `22.20` | `+0.009913` | fail |
| `1` | `3407` | `+0.015359` | `+0.011220` | `+0.00002979` | `0.5380` | `39.60` | `+0.007750` | fail |
| `1` | `3411` | `+0.016165` | `+0.022182` | `+0.00003852` | `0.5360` | `41.40` | `+0.008855` | fail |

Gate interpretation:

- Passing gates: coverage, dSSIM, and worst-regression count.
- Failing gates: mean gain, hard-bottom gain, positive ratio, true-vs-zero,
  true-vs-shuffle, true-vs-normal, and max-outer-worst reporting.
- The locked result is a scientific fail for promotion, not an infra failure:
  all four group jobs exited `rc=0`, generated summaries, and the tmux session
  ended normally.

Synced D9 text evidence includes the locked summary JSON, aggregate CSV,
per-outer CSV, status/log/tmux output, launch script, and compact group-level
logs/summaries. The raw `v37_d9_locked_fixed_policy_selected_actions.csv` and
group selected-action CSVs remain cloud-only runtime artifacts by default.

Decision: `D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING`. Do not promote this sealed
policy from DTA-v3.7 and do not use the locked result to tune thresholds,
features, action membership, checkpoints, or code. Any future DTA work must be a
new train-derived route with locked feedback treated only as final outcome
evidence, not as a development signal.
