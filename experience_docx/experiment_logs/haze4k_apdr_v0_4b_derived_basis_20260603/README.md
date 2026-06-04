# APDR-v0.4B Derived Low-Field Basis Diagnostics

Date: 2026-06-03

Status: completed AutoDL Gate 0 / coefficient-predictability diagnostics.

## Purpose

This diagnostic continues from APDR-v0.4A, where ID embedding passed Gate B
but deployable LowFieldNet, random basis, basis+local, and veil forms failed.
The goal here is not to train a new residual route. It first asks two cheaper
questions:

1. Can low-frequency targets be represented by bases derived from successful
   target fields?
2. Can deployable global image features predict the resulting basis
   coefficients?

Gate C and stop20 remain blocked unless these diagnostics pass.

## Command

Run on `autodl-dehaze3`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603/run_apdr_v0_4b_derived_basis_sigma3.sh
```

Optional quick smoke:

```bash
NUM_IMAGES=64 LOW_SIZES=32 K_VALUES=4,8 bash experience_docx/experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603/run_apdr_v0_4b_derived_basis_sigma3.sh
```

## Expected Artifacts

- `derived_basis_spectrum_sigma3.csv`
- `basis_projection_oracle_sigma3.json`
- `coeff_predictability_cv_sigma3.csv`
- `basis_residual_error_groups.csv`
- `router_overfit32_coeff_vs_field.csv`
- `derived_basis_apdr_v0_4b_derived_basis_sigma3_seed3407.log`
- `basis_projection_meta_rows_sigma3.csv`
- `derived_basis_feature_rows_sigma3.csv`
- `status.txt`

These are text diagnostics only. Do not sync checkpoints, tensor caches, image
outputs, NumPy arrays, or datasets.

## Gate 0

| Metric | Pass line |
| --- | ---: |
| weighted delta L1 drop | `>= 0.60` |
| pred-target corr | `>= 0.75` |
| oracle recovery | `>= 0.50` |
| hard bottom-25% gain | `>= +0.60 dB` |
| easy top-25% gain | `>= -0.010 dB` |
| strong/severe regressions | `0 / 0` |

If projection oracle fails, move to clustered basis banks before any router
training. If projection oracle passes but coefficient predictability is weak,
change router inputs before changing the residual output form.

## Results

Full train-only diagnostic on `3000` images opened `1324` samples under the
frozen train-calibrated correctability threshold.

| low size | K | Gate 0 | L1 drop | Corr | Recovery | Hard gain | Easy gain | Energy |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | fail | `0.4513` | `0.5959` | `0.5616` | `+0.7295 dB` | `+0.0042 dB` | `0.6442` |
| 32 | 8 | fail | `0.5296` | `0.7042` | `0.6540` | `+0.8311 dB` | `+0.0052 dB` | `0.7378` |
| 32 | 16 | pass | `0.6031` | `0.7882` | `0.7344` | `+0.9041 dB` | `+0.0066 dB` | `0.8140` |
| 32 | 32 | pass | `0.6682` | `0.8474` | `0.7996` | `+0.9594 dB` | `+0.0080 dB` | `0.8726` |
| 32 | 48 | pass | `0.7064` | `0.8758` | `0.8323` | `+0.9866 dB` | `+0.0087 dB` | `0.9024` |
| 48 | 16 | pass | `0.6039` | `0.7891` | `0.7361` | `+0.9063 dB` | `+0.0066 dB` | `0.8126` |
| 48 | 32 | pass | `0.6696` | `0.8487` | `0.8027` | `+0.9629 dB` | `+0.0079 dB` | `0.8710` |
| 48 | 48 | pass | `0.7087` | `0.8776` | `0.8366` | `+0.9915 dB` | `+0.0087 dB` | `0.9007` |

Strong-reference and severe regressions were `0 / 0` for the passing rows.

Coefficient predictability is non-degenerate but moderate: mean CV coefficient
corr is about `0.63` at K=4, `0.58` at K=16, and `0.56` at K=32/48, with R2
about `0.38`, `0.32`, and `0.30` respectively. That supports a small
basis-only coefficient router probe, but not local correction yet.

## Decision

Decision label:
`GATE0_PASS_CONTINUE_BASIS_ONLY_COEFF_ROUTER_NO_LOCAL`.

The representation ceiling is strong enough. Do not switch to clustered bases
yet, and do not train dense LowFieldNet or local correction. The next
experiment should train only image-to-coefficients with coefficient loss plus
weighted field loss, prioritizing `low_size=32,K=16` and `low_size=32,K=32`.
