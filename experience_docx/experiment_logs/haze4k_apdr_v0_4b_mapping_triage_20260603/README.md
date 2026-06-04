# APDR-v0.4B-MT Mapping Triage

Date: 2026-06-03

Status: completed on `autodl-dehaze4`; global-stat mapper rescue failed.

## Purpose

This is a post-Gate-C diagnostic for APDR-v0.4B. It does not authorize
local correction, stop20, or a larger router. It separates two possible causes
of the v0.4B mini-val failure:

- global hand-crafted features do not contain enough coefficient information;
- the hidden64 MLP overfits the small train-open coefficient set.

## Command

Run on `autodl-dehaze4`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_mapping_triage_20260603/launch_apdr_v0_4b_mapping_triage_tmux.sh
```

The default run uses:

```text
BASIS_NUM_IMAGES=0
TRAIN_COUNT=128
EVAL_COUNT=256
LOW_SIZE=32
K_VALUES=8,16,32
```

Set `BASIS_NUM_IMAGES=512` or a smaller `K_VALUES` list only for a smoke run.

## Required Outputs

- `mapping_triage_summary_sigma3.json`
- `mapper_family_train128_minival256.csv`
- `coeff_error_by_split_sigma3.csv`
- `coeff_cv_per_component_sigma3.csv`
- `feature_shift_train_vs_minival.csv`
- `open_easy_failure_table_sigma3.csv`
- `mapping_triage_per_image_sigma3.csv`
- `mapping_triage_groups_sigma3.csv`
- `mapping_triage_mlp_history_sigma3.csv`

These are text diagnostics only. Checkpoints, tensors, arrays, datasets, and
image outputs stay out of the repo.

## Results

Default full run completed on `autodl-dehaze4`:

```text
BASIS_NUM_IMAGES=0
TRAIN_COUNT=128
EVAL_COUNT=256
LOW_SIZE=32
K_VALUES=8,16,32
train_open_count=90
mini_val_open_count=114
```

The best mini-val rows did not pass safety:

| Family | Best mini-val pattern | Safety result |
| --- | --- | --- |
| mean coefficient | K8 L1 drop `0.1106`, corr `0.2346`, hard `+0.2097`, easy `+0.0865` | strong/severe `5/7`; unsafe |
| kNN | K8 kNN9 L1 drop `0.0866`, corr `0.2607`, hard `+0.3673`, easy `+0.0481` | strong/severe `6/10`; unsafe |
| PLS | K16 PLS4 corr `0.2540`, hard `+0.4802`, easy `+0.0514` | strong/severe `6/12`; unsafe |
| ridge | K8 ridge10 hard `+0.4807` | L1 drop `-0.1740`, strong/severe `7/24`; unsafe |
| early MLP | K32 hard `+0.4735` | L1 drop `-0.0979`, easy `-0.1786`, strong/severe `9/19`; unsafe |
| zero/no-op | no gain, no regressions | only safety-passing row |

Coefficient-level evidence agrees with the safety result. On mini-val, the best
coefficient correlations stayed low (`~0.28` max), and the lowest normalized
coefficient MSE was the mean-coeff baseline around `0.99`, which did not create
a safe useful field. Per-component CV shows the first few components are
partly predictable, but tails decay quickly: mean component corr is `0.454` for
K8, `0.340` for K16, and `0.265` for K32. Train-vs-mini-val feature distance is
nontrivial (`__all__` mini-val NN distance mean `8.57`, p90 `15.10`).

## Continue Rule

The MT rescue gate failed. Do not continue the v0.4B global-stat mapper, do not
add local correction, and do not run stop20. The next authorized step is a new
APDR-v0.4D preflight that changes the mapper input to frozen ConvIR spatial
features plus confidence/shrinkage, starting with K16 and a spatial-token probe.
