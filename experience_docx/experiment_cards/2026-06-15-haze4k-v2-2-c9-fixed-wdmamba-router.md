# Haze4K v2.2 C9 Fixed-Strong-Expert Baseline + Low-Capacity Group-Min Router

Date: 2026-06-15

Status: `LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER`

## Scope

- Objective: turn the C8-Mini train-derived expert-complementarity evidence into a deployable fixed profile or low-capacity router candidate.
- Runtime host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/`.
- Branch: `codex/haze4k-v2-2-c9-fixed-wdmamba-router`.
- Source C8 evidence: `experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Locked-Test Contract

C9 is train-derived only. It must not read locked per-image outputs, tune from
locked evidence, choose thresholds/features/checkpoints/actions from locked
evidence, distill, train MoE, or run locked Haze4K. The v2.1 locked one-shot is
already consumed and failed; it is evidence-only.

## Fixed Plan

```text
C9-0: provenance, no-locked, metric/script parity, C8 table integrity audit.
C9-A: fixed shrink profile validation.
C9-B: low-capacity router only if C9-A fixed WD0375 is insufficient.
C9-C: group-min shifted validation for the selected C9 candidate.
C10-prep: formal 5x3 plan only if C9-C passes.
```

## Fixed C9-A Profiles

```text
WD0375 = A0 + 0.375 * (WDMamba - A0)
WD050  = A0 + 0.500 * (WDMamba - A0)
FS050  = A0 + 0.500 * (FSNet+UDP - A0)
WD0375_or_FS050_oracle = upper bound only
S3_oracle = upper bound only
```

Primary first candidate: `WD0375`. If it passes C9-A and C9-C, do not train a
router.

## Fixed C9-A Strong Gate

Aggregate:

```text
mean >= +1.50 dB
hard_bottom25 >= +2.00 dB
easy_top25 >= +0.25 dB
positive >= 0.90
dSSIM >= 0
severe <= 36/600
```

Group-min:

```text
each critical bin mean >= +0.50 dB
each critical bin hard >= +0.50 dB
each critical bin positive >= 0.80
each critical bin severe <= 48/600
```

## Conditional C9-B Router

Run only if `WD0375` fails C9-A or C9-C. Candidate actions are fixed before any
router fitting:

```text
A0
WDMamba alpha 0.25 / 0.375 / 0.50
FSNet+UDP alpha 0.25 / 0.50
MB-Taylor alpha 0.0625 / 0.125 / 0.25
```

Allowed router forms:

```text
rule list / monotone thresholds
L2-regularized logistic or multinomial ridge
small calibrated GBDT with max_depth <= 3
```

Forbidden features:

```text
GT PSNR
GT clean-derived test-time features
filename-derived labels
split membership as a prediction feature
locked-derived signals
```

## Required Outputs

C9-0:

- `v22_c9_0_expert_provenance_audit.md`
- `v22_c9_0_no_locked_status.txt`
- `v22_c9_0_render_reproducibility.md`
- `v22_c9_0_metric_parity_report.csv`

C9-A:

- `v22_c9a_fixed_profiles_summary.csv`
- `v22_c9a_fixed_profiles_by_split.csv`
- `v22_c9a_fixed_profiles_groupmin_bins.csv`
- `v22_c9a_fixed_profiles_critical_bin_report.md`
- `v22_c9a_fixed_profiles_decision.md`

C9-B if needed:

- `v22_c9b_router_oof_summary.csv`
- `v22_c9b_router_action_distribution.csv`
- `v22_c9b_router_groupmin_bins.csv`
- `v22_c9b_router_expert_usage_by_group.csv`
- `v22_c9b_router_removal_ablation.csv`
- `v22_c9b_router_feature_ablation.csv`
- `v22_c9b_router_decision.md`

C9-C:

- `v22_c9c_shifted_dimension_summary.csv`
- `v22_c9c_shifted_bin_metrics.csv`
- `v22_c9c_groupmin_decision.md`
- `v22_c9c_bootstrap_wilson_bounds.csv`

Closeout:

- `v22_c9_decision.md`
- `v22_c9_summary.json`
- evidence README update.

## Decision Rules

If `WD0375` passes C9-A and C9-C:

```text
C9A_FIXED_WD0375_STRONG_PASS_AUTHORIZE_C10_FORMAL
```

If `WD0375` fails aggregate, easy/severe, or group-min:

```text
C9A_FIXED_WD0375_FAIL_RUN_C9B_ROUTER
```

If C9-B passes C9-C:

```text
C9B_LOW_CAPACITY_ROUTER_PASS_AUTHORIZE_C10_FORMAL
```

C9 cannot authorize locked test. C10 formal 5x3 must pass first, and even then
only one fixed locked one-shot may be considered.

## Closeout 2026-06-15

C9 ran on `convir-4090` from branch
`codex/haze4k-v2-2-c9-fixed-wdmamba-router` using only the C8 train-derived
per-image tables. It did not rerender locked data, train MoE/router, or run
distillation.

C9-A fixed `WD0375` passed the strong gate:

| Metric | WD0375 |
| --- | ---: |
| mean dPSNR | `+2.512202` |
| hard bottom-25 dPSNR | `+3.505615` |
| easy top-25 dPSNR | `+1.189484` |
| dSSIM | `+0.00167334` |
| positive ratio | `0.973333` |
| severe / 600 | `11.0` |

C9-B was intentionally not run because the fixed `WD0375` profile passed C9-A.
C9-C group-min shifted validation also passed. Worst fixed bins were still above
the predeclared gate: min mean `+1.124603`, min hard `+1.552796`, min positive
`0.900000`, and max severe `40/600`.

C10 formal 5x3 table replay for sealed `WD0375` passed:

- full 600 mean/hard/easy/positive/severe:
  `+2.512202 / +3.505615 / +1.189484 / 0.973333 / 11.0/600`;
- fold-mean mean/hard/easy/positive/severe:
  `+2.516942 / +3.523592 / +1.213647 / 0.973556 / 10.942143/600`;
- fold-worst mean/hard/easy/positive/severe:
  `+2.311024 / +3.347410 / +0.857374 / 0.948276 / 21.818182/600`.

Decision:

```text
C9A_FIXED_WD0375_STRONG_PASS_AUTHORIZE_C10_FORMAL
C10_FORMAL_5X3_WD0375_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW
```

This does not itself run locked Haze4K. It authorizes a separate fixed
one-shot locked replay review for sealed `WD0375` only. Locked output, if run,
must be recorded as evidence and must not tune thresholds, features,
checkpoints, profiles, actions, or distillation targets.

## Locked One-Shot Closeout 2026-06-15

The sealed `WD0375` locked one-shot was consumed once on `convir-4090` from
source commit `1f67309f164733e817bbdc436908e5950fc78ffd`. The command recorded
`one_shot=true` and `no_tuning_from_locked=true`, used alpha `0.375`, and wrote
only text evidence.

Decision:

```text
LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER
```

| Metric | Locked WD0375 |
| --- | ---: |
| count | `1000` |
| mean dPSNR | `+1.442090` |
| hard bottom-25 dPSNR | `+1.529767` |
| easy top-25 dPSNR | `+1.182529` |
| dSSIM | `+0.00247093` |
| positive ratio | `0.938000` |
| nonnegative ratio | `0.938000` |
| severe / 600 | `25.80` |

This locked result is evidence only. It must not be used to tune alpha,
features, checkpoints, profiles, actions, experts, or distillation targets.
Distillation is not performed in this route; any distillation work needs a
separate review and route.
