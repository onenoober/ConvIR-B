# Haze4K DTA-v3.5 FDF-RCS-Lite Route

Date: 2026-06-12

Status: `COMPLETED_RELAXED_FLOW_PASS_STRICT_FAIL_SELECTOR_DIAGNOSTIC_LOCKED_TEST_BLOCKED`

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

## Outcome

The train-derived triage and nested selector postprocess completed on
`convir-4090`. The first queue completed all non-L0 variants but reported
`DTA_V3_5_TRIAGE_QUEUE_FAILED` because the old L0 A0 sanity eval path failed.
After the eval fix was pushed, L0 repair ran with two GPUs from commit `4c7589b`
and completed summary/nested postprocess under the five-GPU follow-up cap.

Completion markers:

```text
train_done_ok=24
eval_done_ok=104
aggregate_done_ok=26
run_ok_markers=26
DTA_V3_5_L0_REPAIR_POSTPROCESS_OK
locked_test_touched=false
```

All L1-L5 variants passed the relaxed diagnostic flow, but none passed the
strict triage gate. Key aggregate results:

| Variant | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst/600 | true-vs-zero | Relaxed | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| L1 `s004_g025` | `+0.071183` | `+0.081919` | `+0.00001536` | `0.6308` | `82.50` | `+0.069518` | pass | fail |
| L2 `s002_g025` | `+0.052706` | `+0.064818` | `+0.00001700` | `0.6250` | `60.50` | `+0.045710` | pass | fail |
| L3 `s004_g015` | `+0.062959` | `+0.075288` | `+0.00001736` | `0.6304` | `69.50` | `+0.056608` | pass | fail |
| L4 `tail_s004_g025` | `+0.066923` | `+0.078181` | `+0.00001603` | `0.6288` | `78.50` | `+0.061267` | pass | fail |
| L5a `res010` | `+0.066249` | `+0.078210` | `+0.00001698` | `0.6296` | `76.25` | `+0.055320` | pass | fail |
| L5b `res015` | `+0.066245` | `+0.078211` | `+0.00001696` | `0.6296` | `76.25` | `+0.055353` | pass | fail |

Interpretation:

- Conservative FDF is the right correction relative to v3.4: positive ratio
  rises from about `0.58-0.59` to about `0.625-0.631`, worst falls from about
  `116-128/600` to about `60-83/600`, and depth attribution remains positive.
- L2 is the safest all-image candidate (`60.5/600` worst) but loses mean and
  attribution surplus; L1 has the best mean and passes the positive-ratio line
  but leaves too much tail risk.
- L4/L5 tail losses and tiny learned residual do not materially improve the
  all-image tail; residual capacity should remain disabled until selector
  quality improves.
- The oracle risk-coverage curve is strong: at `0.50` oracle coverage, non-L0
  variants keep positive mean/hard and zero worst regressions. This confirms
  candidate headroom and moves the bottleneck to deployable selector/calibration.
- The current nested threshold selector is useful but underpowered. Best relaxed
  selector diagnostic is L4 at about `0.21` coverage, selected mean about
  `+0.0197 dB`, selected positive ratio about `0.67`, and worst about
  `31.5/600`; it is not strict-gate or promotion-ready.

Final route decision:

```text
COMPLETED_RELAXED_FLOW_PASS_STRICT_FAIL_SELECTOR_DIAGNOSTIC_LOCKED_TEST_BLOCKED
```

Do not run locked Haze4K test from this route. The next route should improve the
selector/calibration model and risk features rather than increasing router,
FiLM, or residual capacity.

## GPU Use Rule

The first already-running queue may continue as launched. Per the 2026-06-12 user update, all subsequent DTA-v3.5 relaunches or follow-up queues on `convir-4090` cap the launcher to at most five RTX 4090 cards by default (`MAX_GPUS=5`, with `MAX_PARALLEL` not exceeding the capped GPU list).

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
