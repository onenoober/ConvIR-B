# Haze4K v2.3 C11 WD0375-FS050 Two-Profile Selector

Date: 2026-06-15

Status: `PLANNED`

## Scope

- Objective: test whether a minimal train-derived selector can improve on the
  locked-pass `WD0375` baseline by conditionally using `FS050` as a low-risk
  fallback/supplement.
- Runtime host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v23-c11-wd-fs-selector`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_3_c11_wd_fs_selector_20260615/`.
- Branch: `codex/haze4k-v2-3-c11-wd-fs-selector`.
- Source C8 evidence: `experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/`.
- Source C9 evidence: `experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Locked-Test Contract

C11 is train-derived only. It must not read locked per-image outputs, tune from
locked evidence, choose alpha/profile/action/expert features from locked
evidence, run locked Haze4K, or perform distillation. The v2.2 `WD0375` locked
one-shot is evidence-only and cannot be used to alter C11.

## Fixed Candidate Space

Allowed deployable actions:

```text
WD0375 = A0 + 0.375 * (WDMamba - A0)
FS050  = A0 + 0.500 * (FSNet+UDP - A0)
A0     = no-op safety fallback
```

Forbidden in first C11:

```text
WD050
MB-Taylor action
new experts
alpha search beyond the fixed profiles above
patch-level alpha
deep MoE / MLP router
distillation
locked-informed feature/profile/action changes
```

## Fixed Plan

```text
C11-0: route freeze, no-locked status, source artifact manifest, metric parity.
C11-A: WD0375/FS050/A0 oracle decomposition and selected-negative analysis.
C11-B: nested OOF low-capacity selector over WD0375/FS050/A0 only.
C11-C: group-min shifted validation for the best deployable selector.
C11-D: formal 5x3 replay only if C11-C passes.
```

## Selector Policy

Allowed selector forms:

```text
fixed profile baseline
one-stump rule
L2-regularized pairwise FS-vs-WD utility regressor
L2-regularized action utility regressor
```

The action space is fixed to `WD0375`, `FS050`, and `A0`. Features must be
deployable from hazy/A0/expert outputs or train-derived physical proxies already
present in C8 tables. The selector cannot use GT PSNR, GT clean-derived labels,
filename-derived labels, split membership as a prediction feature, or any locked
signal.

## Fixed Gates

Aggregate gate relative to fixed `WD0375`:

```text
mean >= WD0375 + 0.20 dB
hard_bottom25 >= WD0375 - 0.10 dB
easy_top25 >= WD0375 + 0.25 dB
positive >= 0.975
severe <= 11/600
dSSIM >= 0
```

Group-min gate:

```text
min bin mean >= +1.10 dB
min bin hard >= +1.50 dB
min bin positive >= 0.90
max bin severe <= 40/600
```

Formal locked-review authorization requires C11-C pass plus all-seed aggregate
and group-min pass in C11-D. C11-D still does not run locked Haze4K; it can only
authorize a separate review.

## Required Outputs

C11-0:

- `v23_c11_0_route_card.md`
- `v23_c11_0_no_locked_status.txt`
- `v23_c11_0_source_artifact_manifest.json`
- `v23_c11_0_metric_parity_report.csv`

C11-A:

- `v23_c11a_wd_fs_oracle_summary.csv`
- `v23_c11a_wd_fs_unique_wins.csv`
- `v23_c11a_wd_fs_group_composition.csv`
- `v23_c11a_wd_fs_selected_negative_report.md`
- `v23_c11a_wd_fs_decision.md`

C11-B:

- `v23_c11b_selector_oof_summary.csv`
- `v23_c11b_selector_action_distribution.csv`
- `v23_c11b_selector_groupmin_bins.csv`
- `v23_c11b_selector_feature_ablation.csv`
- `v23_c11b_selector_profile_removal_ablation.csv`
- `v23_c11b_selector_decision.md`

C11-C:

- `v23_c11c_shifted_dimension_summary.csv`
- `v23_c11c_shifted_bin_metrics.csv`
- `v23_c11c_groupmin_decision.md`
- `v23_c11c_bootstrap_wilson_bounds.csv`

C11-D:

- `v23_c11d_formal_5x3_summary.csv`
- `v23_c11d_formal_groupmin_summary.csv`
- `v23_c11d_formal_selector_stability.csv`
- `v23_c11d_formal_decision.md`

Closeout:

- `v23_c11_decision.md`
- `v23_c11_summary.json`
- evidence README update.

## Decision Rules

If C11-A oracle has insufficient FS050 unique headroom:

```text
C11A_ORACLE_HEADROOM_FAIL_STOP_SELECTOR
```

If the best C11-B selector does not clear aggregate gate:

```text
C11B_SELECTOR_AGGREGATE_FAIL_RUN_C11C_FOR_FAILURE_LOCALIZATION
```

If C11-C fails aggregate or group-min:

```text
C11C_GROUPMIN_OR_AGGREGATE_FAIL_NO_FORMAL_LOCKED
```

If C11-D passes:

```text
C11D_FORMAL_5X3_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW
```

Otherwise locked remains blocked and the likely next route is C12 `WD0375`
distillation feasibility.
