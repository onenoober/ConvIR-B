# Haze4K DTA-v3.7 U-TQS-Mix Evidence

Date: 2026-06-13

Status: `PHASE_B2_ENRICHED_TABLE_POLICY_STRICT_FAIL_REAL_BLEND_NEXT`

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
- Locked test: untouched and blocked unless formal strict pass seals a fixed policy.

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
