# Haze4K DTA-v3.4 FDF-TSR Fine-Tune Route

Date: 2026-06-12

Status: `PLANNED_USER_EXPLICIT_TEST_OVERRIDE_ONE_SHOT`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Family: Depth-transmission adapters.
- Route name: DTA-v3.4 FDF-TSR, Feature-level Depth Fusion + Tail-Safe Residual Selector.
- Branch: `codex/haze4k-dta-v3-4-fdf-tsr-finetune`.
- Runtime host order for this route: try `convir-5090` first, then fall back to `convir-4090` only if `convir-5090` is unavailable.
- Primary runtime workspace: `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune`.
- Primary Python: `/home/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`.
- Data: `/home/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/home/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/home/caozhiyang/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- Evidence root: `experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611/`.

## User Override

The user explicitly requested one Haze4K test experiment and result images with the gate set as loose as possible. This route records that request as:

```text
USER_EXPLICIT_TEST_OVERRIDE_ONE_SHOT
```

This does not unlock repeated test tuning. The test output from this route must not be used to select another checkpoint, threshold, gate, residual scale, or route variant. Any follow-up must return to train-derived validation or nested calibration unless the user gives another explicit test command.

## Fine-Tuning Rule

Default for this repository is now fine-tuning from the official Haze4K checkpoint. DTA-v3.4 starts from `haze4k-base.pkl` with partial loading of new `DTA.*` modules; it is not a from-scratch route.

## Hypothesis

DTA-v3.3 showed strong depth-guided residual signal but unsafe worst-tail behavior, while the implemented RouterFusion gate over-suppressed useful action and still leaked tail failures. DTA-v3.4 moves most depth use to feature-level fusion and leaves the late RGB correction as no/near-zero physical action plus a tiny bounded learned residual.

## Architecture Change

New DTA-v3.4 additions under `DTA.*`:

- stage2 feature-depth fusion module and scalar gate;
- stage3 feature-depth fusion module and scalar gate;
- final decoder feature-depth fusion module and scalar gate;
- optional tiny SafeMix learned residual with `phys_weight=0` for the quick test path.

The quick route uses wide gates by user request:

```text
DTA feature gate limit = 1.0
DTA feature gate bias = 3.0
stage2/stage3 DTA gate limit = 1.0
late physical RGB delta = disabled or near zero
```

## Partial Load And Initialization

- Load official A0 checkpoint with `--init_model_partial` and `--partial_new_prefixes DTA.`.
- Official ConvIR-B keys must match shape and load from A0.
- New DTA modules are zero-output initialized, so the route starts from an A0-equivalent no-op before fine-tuning.
- Trainable scopes:
  - `dta_fdf_feature_only`: feature-fusion modules, `log_alpha`, transmission and uncertainty heads.
  - `dta_fdf_tsr_residual`: feature-fusion modules plus SafeMix learned residual/gate.
  - `dta_fdf_tsr_full`: adds RouterFusion heads for diagnostic use only.

## Quick Candidate

The first quick user-requested test candidate is:

```text
E2 = feature-level depth fusion + tiny bounded learned residual
train_scope = dta_fdf_tsr_residual
phys_weight = 0.0
learned_weight = 1.0
safe_mix_delta_clip = 0.03
feature_fusion_strength = 0.12
feature_fusion_gate_bias = 3.0
```

## Required Evidence

- `run_dta_v3_4_fdf_tsr_convir5090.sh`
- `dta_v3_4_fdf_tsr_summary.json/csv`
- `train_eval_depth_matrix_v34_fdf_tsr_*_fallback_test.json/csv`
- `r0_vs_rdepth_attribution_v34_fdf_tsr_*_fallback_test.csv`
- test contact-sheet manifest and cloud image paths
- local copied result PNGs for user viewing, not committed

## Stop Rule

After the one-shot test and result images, stop and sync text evidence. Do not run additional Haze4K test variants from the test result. If the result suggests follow-up, design that follow-up on train-derived validation first.
