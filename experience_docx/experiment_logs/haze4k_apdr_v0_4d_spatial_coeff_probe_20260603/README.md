# APDR-v0.4D Spatial Coefficient Probe

Date: 2026-06-03

Status: completed on `autodl-dehaze4`; base spatial probe failed the promotion
gate, while confidence/no-op fallback remains diagnostic-only.

## Purpose

This preflight follows APDR-v0.4B-MT. The MT diagnostic showed that global
hand-crafted statistics do not safely rescue coefficient mapping. This probe
tests the next cheapest question before writing a full spatial router:

```text
Do frozen ConvIR multi-scale spatial features improve basis coefficient mapping
on train128/mini-val256 compared with global statistics alone?
```

It does not train ConvIR-B, does not train APDR residuals, does not write a
checkpoint, and does not authorize stop20.

## Command

Run on `autodl-dehaze4`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603/launch_apdr_v0_4d_spatial_coeff_probe_tmux.sh
```

Default full run:

```text
BASIS_NUM_IMAGES=0
TRAIN_COUNT=128
EVAL_COUNT=256
LOW_SIZE=32
K_VALUES=16,32
SPATIAL_GRID=4
SPATIAL_PROJ_CHANNELS=8
```

Post-run confidence/no-op fallback sweep:

```bash
python3 experience_docx/tools/analyze_haze4k_apdr_v0_4d_spatial_confidence_sweep.py \
  --per_image_csv experience_docx/experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603/spatial_coeff_probe_per_image_sigma3.csv \
  --output_dir experience_docx/experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603 \
  --label sigma3
```

## Outputs

- `spatial_coeff_probe_summary_sigma3.json`
- `spatial_coeff_probe_mapper_family_sigma3.csv`
- `spatial_coeff_probe_coeff_error_by_split_sigma3.csv`
- `spatial_coeff_probe_groups_sigma3.csv`
- `spatial_coeff_probe_open_easy_failure_sigma3.csv`
- `spatial_coeff_probe_per_image_sigma3.csv`
- `spatial_coeff_probe_confidence_sweep_sigma3.csv`
- `spatial_coeff_probe_confidence_sweep_summary_sigma3.json`
- `spatial_coeff_probe_apdr_v0_4d_spatial_coeff_probe_sigma3_seed3407.log`
- `status.txt`
- `tmux_exit_apdr_v04d_spatial_probe_full_20260603.txt`

## Run Status

The full AutoDL run completed successfully:

```text
exit_code=0
start_v0_4d_spatial_coeff_probe ... 2026-06-03T19:42:18+08:00
complete_v0_4d_spatial_coeff_probe apdr_v0_4d_spatial_coeff_probe_sigma3_seed3407 2026-06-03T20:06:02+08:00
```

Full-run split and feature dimensions:

```text
train_open_count=90
mini_val_open_count=114
feature_dims: global=73, spatial_priors=240, convir_spatial=1440, global_plus_spatial=1753
```

## Base Probe Results

The base spatial-feature probe did improve some mini-val gain/correlation rows,
but every nonzero mapper still failed the safety gate. The only safe rows were
zero/no-op rows.

| Mini-val row | L1 drop | Corr | Mean gain | Hard gain | Easy gain | Strong/severe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `spatial_priors_kernel_knn_9`, K16 | `0.1331` | `0.2699` | `+0.2361` | `+0.4108` | `+0.0127` | `3/5` |
| `convir_spatial_kernel_knn_9`, K16 | `0.1315` | `0.3054` | `+0.2724` | `+0.5211` | `+0.0621` | `4/6` |
| `global_plus_spatial_kernel_knn_9`, K16 | `0.1327` | `0.3048` | `+0.2784` | `+0.5004` | `+0.0687` | `4/6` |
| `convir_spatial_pls_16`, K16 | `0.0781` | `0.3448` | `+0.3525` | `+0.6571` | `+0.0969` | `7/11` |
| `global_zero_field`, K16 | `0.0000` | none | `+0.0000` | `+0.0000` | `+0.0000` | `0/0` |

Summary gate:

```text
safe_count=8
candidate_count=88
best_safe_l1=global_zero_field / no-op
```

All 8 safe rows are the feature-set/K no-op baselines. Therefore frozen spatial
features alone do not authorize a real v0.4D router, stop20, or local
correction.

## Confidence Sweep

The post-run confidence sweep simulates a target-free no-op fallback: rows below
a confidence threshold are treated as anchor/no-op output. It uses only the
existing per-image table and does not rerun the model.

Best same-split diagnostic rows:

| Sweep row | Confidence key | Keep count | L1 drop | Mean gain | Hard gain | Easy gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_plus_spatial_kernel_knn_9`, K16 | `pred_abs_mean >= 0.0101` | `23/128` | `0.1207` | `+0.1541` | `+0.4242` | `+0.0618` | `0/0` |
| `spatial_priors_ridge_10`, K16 | `pred_abs_mean >= 0.0139` | `39/128` | `0.0719` | `+0.2757` | `+0.5273` | `+0.1587` | `1/0` |
| `spatial_priors_kernel_knn_9`, K16 | `confidence_proxy >= 0.0844` | `46/128` | `0.0755` | `+0.1002` | not primary | `+0.0108` | `0/0` |

This is encouraging only as a diagnostic: the thresholds were selected on the
same mini-val table, so they do not prove deployable generalization. They do
show that confidence/shrinkage is the only remaining APDR-v0.4D subroute with
some evidence value.

## Decision

Decision label:

```text
SPATIAL_PROBE_FAIL_CONFIDENCE_DIAGNOSTIC_ONLY
```

Do not run stop20, do not add local correction, and do not start a full
spatial-router training scout from this evidence. The only reasonable follow-up
is an independent confidence/no-op fallback confirmation on a fixed held-out
split or alternate seed. Promotion requires a nonzero target-free confidence
rule to keep severe `0`, strong `<=1`, easy `>=-0.02`, and retain useful gain
without threshold selection on the reported mini-val table.
