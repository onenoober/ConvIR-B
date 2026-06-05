# ConvIR-Dehaze-v1.7-RCExpertMix

Date: 2026-06-05

Status: completed cloud intermediate-analysis route. The A0+UDP alpha-mixture
oracle passed the mechanism gate, but the deployable full-train OOF policy and
train-derived heldout confirmation failed the written gates. Locked Haze4K
test remains blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: A0-fallback risk-controlled expert mixture.
- Baseline expert `E0`: official ConvIR-B A0.
- Hard expert `E1`: official UDPNet ConvIR checkpoint from v1.5 Phase 0.
- Candidate output:

```text
Y_alpha = A0 + alpha * (UDPNet - A0)
alpha in {0.00, 0.25, 0.50, 0.75, 1.00}
```

- Router: low-capacity gain/risk heads plus OOD veto. Default is `alpha=0`
  A0 fallback.
- Execution environment: cloud server `dehaze1`; local WSL checkout is editing
  and compile/syntax-only.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/`.
- Branch or isolated workspace:
  `codex/haze4k-v1-7-risk-controlled-expert-mix`.
- Locked Haze4K test policy: blocked in this route until OOF and heldout gates
  pass. Do not use v1.6 locked-test per-image results for threshold, feature,
  expert, or checkpoint selection.

## Motivation

v1.6 closed the fixed single-threshold A0+UDP switch as
`LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`. The result did not invalidate the
expert architecture: the A0+UDP oracle upper bound was strong, and internal
true OOF switch evidence passed, but the fixed single-feature policy did not
generalize to locked test.

v1.7 keeps the expert bank fixed and changes only the calibration/policy:

```text
600-image internal calibration -> full 3000-image train-derived calibration
hard switch -> shrink/mix alpha policy
single threshold -> gain/risk heads + OOD veto
OOF only -> OOF plus train-derived heldout confirmation
```

## Intermediate Outputs

The route must produce these before any locked-test discussion:

```text
v17_fulltrain_a0_udp_feature_table.csv
v17_fulltrain_a0_udp_feature_summary.json
v17_oracle_switch_mix_alpha_grid.json
v17_oof_gain_risk_predictability.csv
v17_oof_risk_coverage_curves.csv
v17_policy_stability_by_fold.csv
v17_calibration_curve_bad_risk.csv
v17_trainheldout_confirm_summary.json
v17_analysis_status.json
```

The feature extraction step is GPU/runtime work. The risk-control analysis is
table-only and can be rerun independently after the feature table exists.

## Calibration Source

Use the existing train-derived split JSON:

```text
experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json
```

The v1.7 full-train calibration set is the union:

```text
train_inner: 2400
val_regular: 300
val_hard: 300
total: 3000
```

The heldout confirmation fits/selects on `train_inner` and confirms on
`val_regular + val_hard`.

## Gate 1: Oracle / Mechanism

Purpose: verify the fixed A0+UDP bank still has useful alpha-mix upper bound.

Pass if the GT oracle alpha-grid summary has:

```text
mean_delta >= +0.30 dB
hard_bottom25_delta >= +0.50 dB
easy_top25_delta >= 0.00 dB
worst_ratio == 0
```

This gate is mechanism evidence only; it does not authorize locked test.

## Gate 2: Full-Train OOF Internal Gate

Purpose: decide whether the risk-controlled policy is strong enough to deserve
heldout confirmation.

Pass line:

```text
OOF combined mean_delta >= +0.25 dB
OOF hard_bottom25_delta >= +0.55 dB
OOF easy_top25_delta >= +0.03 dB
OOF SSIM_delta >= 0
OOF worst_ratio <= 0.035
OOF strong_ratio <= 0.08
all folds Utility Gate pass
at least 4/5 folds Promotion-style gate pass
fixed policy parameters stable across folds
```

## Gate 3: Train-Derived Heldout Confirmation

Purpose: avoid touching locked test when OOF strength does not transfer to an
untouched train-derived confirmation set.

Fit/select policy on `train_inner` and confirm on `val_regular + val_hard`.

Pass line:

```text
heldout mean_delta >= +0.18 dB
heldout hard_bottom25_delta >= +0.35 dB
heldout easy_top25_delta >= -0.02 dB
heldout worst_ratio <= 0.04
heldout strong_ratio <= 0.10
```

Only if Gate 2 and Gate 3 both pass may a later immutable one-shot locked-test
command be written. This route card alone does not authorize locked test.

## Cloud Run Contract

- Remote workspace:
  `/root/autodl-tmp/workspace/ConvIR-B-v1-7-rcmix-runtime`.
- Python: `/root/miniconda3/envs/convir-cu128/bin/python`.
- Data root: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`.
- A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Official UDPNet checkpoint:
  `/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt`.
- Split JSON:
  `experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json`.
- Splits: `train_inner`, `val_regular`, `val_hard`.
- Runtime launcher:
  `experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/run_v17_intermediate_analysis.sh`.

## Stop Rules

- If feature extraction fails from missing data/checkpoint/cache/imports, mark
  `PREFLIGHT_FAILED_ENGINEERING` or `FAILED_INFRA`; do not reinterpret it as a
  scientific route result.
- If full-train OOF fails, do not run heldout-selected locked test.
- If heldout confirmation fails, do not touch locked test.
- Do not add FAM2/APDR or change the expert bank in v1.7A.
- Do not tune any v1.7 feature, threshold, policy, or alpha from v1.6 locked
  per-image results.

## Cloud Result

Cloud run source snapshot: `8f3d631` copied to
`/root/autodl-tmp/workspace/ConvIR-B-v1-7-rcmix-runtime`.

Feature extraction completed on `dehaze1` with `3000` train-derived rows:
`train_inner=2400`, `val_regular=300`, `val_hard=300`. The full feature table
is:

```text
experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_fulltrain_features/v17_fulltrain_a0_udp_feature_table.csv
```

The order-fixed table-only analysis was rerun on cloud and wrote:

```text
experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_mix_analysis/v17_analysis_status.json
experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_mix_analysis_orderfix_rerun.log
```

Locked Haze4K test touched: `false`.

Gate 1 oracle/mechanism passed. The GT oracle alpha mix reached mean
`+0.8689 dB`, hard bottom-25 `+0.9623 dB`, easy top-25 `+0.8245 dB`,
SSIM `+0.000283`, worst ratio `0`, and strong ratio `0`. This confirms that
the frozen A0+UDP expert bank still has a large upper bound.

Fixed global shrinkage was informative but not safe enough for promotion:
`alpha=0.25` reached mean `+0.3723 dB` and easy top-25 `+0.5182 dB` but had
worst ratio `0.109`; `alpha=0.50` reached mean `+0.4581 dB` and hard bottom-25
`+0.3905 dB` but had worst ratio `0.215`; `alpha=1.00` remained unsafe as a
global replacement with mean `-0.2430 dB`, easy top-25 `-0.7218 dB`, and worst
ratio `0.4907`.

Gate 2 full-train OOF failed. The selected low-capacity gain/risk/OOD policy
used `tau_gain=0.45`, `tau_risk=0.20`, and `tau_ood_quantile=1.0`, with
coverage `0.1557`. Its combined OOF result was mean `+0.1079 dB`, hard
bottom-25 `+0.1417 dB`, easy top-25 `+0.1020 dB`, SSIM `+0.000054`, worst
ratio `0.0067`, and strong ratio `0.0107`. It failed the preregistered OOF
mean and hard-gain margins (`+0.25` and `+0.55` respectively), and fold utility
pass count was `0/5`.

Gate 3 train-derived heldout confirmation failed. Fitting/selecting on
`train_inner` and confirming on `val_regular + val_hard` produced coverage
`0.1117`, mean `+0.0945 dB`, hard bottom-25 `+0.1297 dB`, easy top-25
`+0.0597 dB`, SSIM `+0.000041`, worst ratio `0.0033`, and strong ratio
`0.0282`. It missed the heldout mean and hard-gain lines (`+0.18` and `+0.35`).

The current v1.7A conclusion is that shrink/mix improves the expert-bank
mechanism and reduces tail risk versus hard UDP replacement, but the tested
low-capacity deployable router is not strong enough to authorize locked-test
confirmation.

## Current Decision State

Decision label: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`.

Do not tune the current v1.7A thresholds, OOD cutoff, feature set, alpha set, or
low-capacity heads from these results and call it the same route. Any follow-up
must be a new predeclared router/calibration route and must pass OOF plus
train-derived heldout before locked Haze4K test is considered.
