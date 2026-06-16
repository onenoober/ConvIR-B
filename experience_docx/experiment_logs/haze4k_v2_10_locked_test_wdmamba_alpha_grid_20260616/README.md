# Haze4K v2.10 Locked WDMamba Alpha Grid

Decision: `V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

This route evaluates the predeclared WDMamba residual-shrinkage alpha grid on the Haze4K locked test split (`1000` images). It is a diagnostic audit only and must not be used to select or retune alpha.

Metric protocol: v2.2 locked one-shot compatible. A0 and alpha candidates use the same A0 32-pad SSIM convention as v2.2 WD0375; WDMamba standalone endpoint SSIM is recorded separately and parity-matches v2.2.

## Absolute Metrics

| alpha | label | PSNR | SSIM grid32 | WDMamba endpoint SSIM | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 | worst dPSNR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | A0 / ConvIR-B | 34.145502 | 0.989619 |  | +0.000000 | +0.000000 | +0.000000 | 0.000000 | 0.00 | +0.000000 |
| 0.125 | alpha=0.125 | 34.675277 | 0.990609 |  | +0.529775 | +0.505304 | +0.507815 | 0.957000 | 6.60 | -1.227917 |
| 0.250 | alpha=0.250 | 35.163759 | 0.991433 |  | +1.018257 | +1.017456 | +0.915022 | 0.949000 | 15.00 | -2.619633 |
| 0.375 | WD0375 | 35.587591 | 0.992090 |  | +1.442090 | +1.529767 | +1.182529 | 0.938000 | 25.80 | -3.985588 |
| 0.500 | alpha=0.500 | 35.920460 | 0.992578 |  | +1.774958 | +2.030727 | +1.279034 | 0.913000 | 35.40 | -5.255131 |
| 0.750 | alpha=0.750 | 36.203139 | 0.993026 |  | +2.057637 | +2.888437 | +0.939985 | 0.838000 | 76.20 | -7.464008 |
| 1.000 | WDMamba full | 35.917147 | 0.992711 | 0.992468 | +1.771646 | +3.314757 | +0.052429 | 0.729000 | 144.00 | -9.413492 |

## Reliability Checks

- Pair count: `1000` locked-test images.
- v2.2 parity pass: `True` with matched count `1000`.
- Max abs parity diffs: A0 PSNR `0.0`, A0 SSIM `0.0`, WD0375 PSNR `0.0`, WD0375 SSIM `0.0`, WDMamba PSNR `0.0`, WDMamba SSIM `0.0`.
- Preliminary v28/NH metric reuse output was discarded before sync because A0/WD0375 SSIM did not match v2.2; this directory contains only the corrected v2.2-compatible rerun.
- Locked policy: diagnostic-only; no locked-grid alpha selection or retuning.

## Evidence Files

- `v210_haze4k_locked_wdmamba_alpha_grid_absolute_metrics.csv`
- `v210_haze4k_locked_wdmamba_alpha_grid_compact_metrics.csv`
- `v210_haze4k_locked_wdmamba_alpha_grid_per_image.csv`
- `v210_haze4k_locked_wdmamba_alpha_grid_summary.json`
- `v210_parity_with_v22_wd0375.json`
- `commands/run_v210_locked_alpha_grid_parallel.sh`
- `runtime_logs/`
