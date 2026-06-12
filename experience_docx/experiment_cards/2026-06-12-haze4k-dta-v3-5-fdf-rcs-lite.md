# Haze4K DTA-v3.5 FDF-RCS-Lite Route

Date: 2026-06-12

Status: `PLANNED_RELAXED_TRAIN_DERIVED_FLOW`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Family: Depth-transmission adapters.
- Route name: DTA-v3.5 FDF-RCS-Lite, Feature-level Depth Fusion + Risk-Calibrated Selector Lite.
- Branch: `codex/haze4k-dta-v3-5-fdf-rcs-lite`.
- Anchor commit: `2d529d4` from `github/codex/haze4k-official-arch-anchor`.
- Continuation base: DTA-v3.4 FDF-TSR commit `c742fe4`; this branch remains anchor-descended and explicitly continues v3.4 because v3.4 already proved feature-level depth attribution while failing tail safety.
- Runtime host: `convir-4090` only.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612/`.

## Locked-Test Policy

Locked Haze4K test remains blocked for this route. The user asked to relax stage-continuation metrics so the train-derived diagnostic flow can complete; this card interprets that as permission to continue train-derived triage and nested calibration even if strict triage gates fail, not as permission to tune or rerun locked test.

## Why This Route Exists

DTA-v3.4 proved that feature-level depth fusion (FDF) makes real depth matter: true-vs-zero/shuffle/normal surplus was large on train-derived folds. It also failed because feature action was too broad and there was no nested risk-calibrated selector. DTA-v3.5 keeps the FDF mechanism but reduces action budget and adds explicit A0/no-regression and action-budget losses, then performs nested selector calibration from OOF evidence.

Mechanism sentence:

```text
If we reduce FDF gate/action budget and calibrate a selector on train-derived OOF tables, positive ratio and worst-tail behavior should improve because depth action becomes conservative and false interventions can fall back to A0.
```

## Architecture And Loss Changes

New code on top of v3.4:

- records `gate * delta` feature-action tensors and `*_feature_action_abs_mean` stats for stage2, stage3, and final FDF;
- adds differentiable feature gate/action budget losses from DTA aux state;
- adds safe residual gate/action budget losses for tiny learned residual variants;
- adds explicit A0 MSE no-regression loss `--dta_ref_mse_regression_weight` / `--dta_ref_mse_regression_margin`;
- extends eval per-image CSVs with transmission GT summaries when `--include_trans_stats` is set;
- adds v3.5 triage summary and nested selector tools.

Partial-load contract:

- load official A0 with `--init_model_partial --partial_new_prefixes DTA.`;
- all original ConvIR-B keys must load by shape;
- new DTA modules are zero-output initialized as in v3.4;
- FDF gate bias is negative in v3.5, so loaded models start near A0/no-op and train with low action coverage.

## Initial Train-Derived Matrix

The initial relaxed queue runs 2 folds x 2 seeds on `convir-4090` and never touches locked test.

| ID | Variant | Scope | Key config | Purpose |
| --- | --- | --- | --- | --- |
| L0 | `l0_a0_sanity` | no train | A0 vs A0 eval | metric/split sanity |
| L1 | `l1_fdf_lite_s004_g025_bm2` | `dta_fdf_feature_only` | strength 0.04, gate 0.25, bias -2 | conservative FDF baseline |
| L2 | `l2_fdf_lite_s002_g025_bm2` | `dta_fdf_feature_only` | strength 0.02, gate 0.25, bias -2 | smaller action budget |
| L3 | `l3_fdf_lite_s004_g015_bm2` | `dta_fdf_feature_only` | strength 0.04, gate 0.15, bias -2 | lower gate limit |
| L4 | `l4_fdf_lite_tail_s004_g025_bm2` | `dta_fdf_feature_only` | L1 + stronger A0/CVaR/group/patch/action losses | tail-preservation pressure |
| L5a | `l5_res010_tail_s004_g025_bm2` | `dta_fdf_tsr_residual` | L4 + learned residual clip 0.010 | tiny residual audit |
| L5b | `l5_res015_tail_s004_g025_bm2` | `dta_fdf_tsr_residual` | L4 + learned residual clip 0.015 | tiny residual audit |

## Loss Weights

The first L4/L5 tail-safe settings use:

```text
lambda_ref_l1        = 0.012
lambda_ref_mse       = 0.012
lambda_cvar          = 0.012
lambda_group         = 0.010
lambda_patch_ssim    = 0.008
lambda_feature_gate  = 0.003
lambda_feature_action= 0.006
lambda_safe_gate     = 0.002 for L5 only
lambda_safe_action   = 0.004 for L5 only
```

## Required Evidence

- `run_dta_v3_5_fdf_rcs_lite_convir4090.sh`
- `launch_dta_v3_5_fdf_rcs_lite_triage_convir4090.sh`
- `dta_v3_5_fdf_rcs_triage_summary.json/csv`
- `dta_v3_5_fdf_rcs_triage_variant_summary.csv`
- `v35_oof_per_image_action_table.csv`
- `v35_oracle_risk_coverage_curve.csv`
- `v35_selector_nested_calibration_report.json/csv`
- `v35_selector_nested_selected_images.csv`
- per-run train/eval/aggregate logs and depth-control matrices.

## Relaxed Flow Rule

Strict promotion gates remain recorded, but the run queue uses relaxed flow gates so L1-L5 and nested calibration complete even if early strict gates fail. Relaxed flow pass is diagnostic only; it does not authorize locked test or promotion.

## Decision Logic

```text
if oracle risk-coverage at coverage >= 0.50 passes tail:
    candidate has selector headroom; improve selector/calibration.
elif oracle only passes below coverage 0.25:
    candidate action remains too risky; lower strength/gate or change fusion point.
elif lite-FDF improves positive ratio/worst while mean drops:
    conservative action direction is correct; continue selector.
elif residual variants increase mean but worsen tail:
    keep residual disabled and continue feature-only.
```
