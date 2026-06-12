# Haze4K DTA-v3.4 FDF-TSR Fine-Tune Route

Date: 2026-06-12

Status: `TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED`

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
  - `dta_fdf_tsr_plus_film`: adds stage2/stage3 FiLM-style DTA modules for the high-capacity triage candidate.

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
- train-derived triage summary: `dta_v3_4_fdf_tsr_triage_summary.json/csv`
- train-derived variant summary: `dta_v3_4_fdf_tsr_triage_variant_summary.csv`

## Stop Rule

After the one-shot test and result images, stop and sync text evidence. Do not run additional Haze4K test variants from the test result. If the result suggests follow-up, design that follow-up on train-derived validation first.

## 2026-06-12 One-Shot Haze4K Test Result

Status: `COMPLETED_ONE_SHOT_TEST_FAIL_NO_FURTHER_TEST_SELECTION`.

The user-requested `convir-5090` run completed for `E2=e2_tiny_residual` with widest gates and fallback-A test evaluation. The run first attempted to build the Depth Anything cache on `convir-5090`, but Hugging Face access failed with `Network is unreachable`; the existing 4000-file Depth Anything cache was copied from `convir-4090` and the run continued on `convir-5090`.

Configuration:

```text
train_scope = dta_fdf_tsr_residual
train data = Haze4K train, full 3000 images
stage = quick5full
checkpoint init = official Haze4K A0
feature_fusion_strength = 0.12
feature_fusion_gate_limit = 1.0
feature_fusion_gate_bias = 3.0
safe_mix_delta_clip = 0.03
safe_mix_phys_weight = 0.0
safe_mix_learned_weight = 1.0
late physical RGB action = disabled
```

Haze4K test fallback-A depth-control matrix:

| eval depth | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst <= -0.20 | strong <= -0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true/invert | `-0.014802` | `+0.039381` | `+0.00004687` | `0.489` | `257/1000` | `122/250` |
| zero | `-0.107865` | `-0.011653` | `-0.00009474` | `0.358` | `355/1000` | `156/250` |
| shuffle | `-0.120837` | `-0.019604` | `-0.00008462` | `0.306` | `343/1000` | `150/250` |
| normal | `-0.153693` | `-0.040721` | `-0.00013619` | `0.254` | `395/1000` | `163/250` |

Depth attribution on locked test is positive (`true-vs-zero=+0.093063`, `true-vs-shuffle=+0.106035`, `true-vs-normal=+0.138891`), but the model is not promotion-ready because absolute mean is negative, positive ratio is below 0.50, and worst regressions are very high. The one-shot test result must not be used to select another test-set variant.

Result images were copied to a local non-repo folder for user inspection:

```text
/home/ubuntu/workspace/dta_v3_4_fdf_tsr_test_visuals_20260612/
```

## 2026-06-12 Train-Derived Triage Outcome

Status: `TRIAGE_GATE_FAIL_LOCKED_TEST_BLOCKED`.

After the one-shot test was archived, the route returned to train-derived
validation. `convir-5090` completed the predeclared low-cost triage:

```text
variants = e1_feature_only, e2_tiny_residual, e3_tsr_full, e4_plus_film
folds = fold0, fold1
seeds = 3407, 3411
stage = quick5full
RUN_TEST = 0
locked_test_touched = false
```

Aggregate train-derived fallback-A results:

| Variant | mean dPSNR | hard bottom-25 | dSSIM | positive ratio | worst <= -0.20 | true-vs-zero | true-vs-shuffle | true-vs-normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e1_feature_only` | `+0.079169` | `+0.053351` | `+0.00000911` | `0.5796` | `116.25/600` | `+0.249893` | `+0.211476` | `+0.247717` |
| `e2_tiny_residual` | `+0.080316` | `+0.054531` | `+0.00001041` | `0.5788` | `115.75/600` | `+0.255835` | `+0.213484` | `+0.248146` |
| `e3_tsr_full` | `+0.075988` | `+0.051520` | `+0.00000419` | `0.5804` | `127.75/600` | `+0.244546` | `+0.204833` | `+0.237288` |
| `e4_plus_film` | `+0.081446` | `+0.057209` | `+0.00000786` | `0.5896` | `123.00/600` | `+0.216621` | `+0.184871` | `+0.208891` |

The triage confirms that FDF-TSR produces strong train-derived mean, hard, and
depth-control surplus, but no variant passes the written safety gate because
positive ratio remains below `0.630` and worst regressions remain far above
`48/600` (`max_run_worst` is `137-139` depending on variant). Formal
5-fold x 3-seed validation is therefore blocked, and no additional Haze4K test
variant is allowed from these results.

Primary triage files:

- `dta_v3_4_fdf_tsr_triage_summary.json`
- `dta_v3_4_fdf_tsr_triage_summary.csv`
- `dta_v3_4_fdf_tsr_triage_variant_summary.csv`
- `train_eval_depth_matrix_v34_fdf_tsr_*_fallback_train.json/csv`
- `r0_vs_rdepth_attribution_v34_fdf_tsr_*_fallback_train.csv`
