# Haze4K v2.10 Locked WDMamba Alpha Grid

Date: 2026-06-16

Status: `COMPLETED_GATE_PASS`

## Purpose

Report the predeclared WDMamba residual-shrinkage alpha grid on the Haze4K
locked test split (`1000` images), using the same A0 and WDMamba checkpoints as
the v2.2 locked WD0375 one-shot.

This is a locked-test diagnostic audit only. It extends the already consumed
v2.2 one-shot by listing the fixed alpha grid, but it must not be used to select
a new alpha, retune WD0375, change checkpoints, or replace the v2.2 locked-pass
narrative.

## Fixed Protocol

```text
candidate(alpha) = A0 + alpha * (WDMamba - A0)
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
```

## Data And Runtime

- Runtime host: `convir-4090`
- Remote workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router-locked`
- Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Dataset:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K/test`
- A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- WDMamba checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`

The run used 8 GPU shards. Each image was evaluated once for A0 and once for
WDMamba, then all 7 alpha candidates were computed from the same outputs.

## Metric Protocol

The final run uses the v2.2 locked one-shot metric convention. A0 and alpha
candidates use the same A0 32-pad SSIM convention as v2.2 WD0375. The standalone
WDMamba endpoint SSIM is also recorded separately and matches the v2.2 endpoint.

A preliminary reuse of the NH-HAZE v2.8 evaluator was discarded before sync
because A0/WD0375 SSIM did not parity-match v2.2; only the corrected
v2.2-compatible run is archived.

## Result

Decision: `V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

| alpha | PSNR | SSIM grid32 | WDMamba endpoint SSIM | mean dPSNR | hard dPSNR | easy dPSNR | positive | severe/600 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 34.145502 | 0.989619 |  | +0.000000 | +0.000000 | +0.000000 | 0.000000 | 0.00 |
| 0.125 | 34.675277 | 0.990609 |  | +0.529775 | +0.505304 | +0.507815 | 0.957000 | 6.60 |
| 0.250 | 35.163759 | 0.991433 |  | +1.018257 | +1.017456 | +0.915022 | 0.949000 | 15.00 |
| 0.375 | 35.587591 | 0.992090 |  | +1.442090 | +1.529767 | +1.182529 | 0.938000 | 25.80 |
| 0.500 | 35.920460 | 0.992578 |  | +1.774958 | +2.030727 | +1.279034 | 0.913000 | 35.40 |
| 0.750 | 36.203139 | 0.993026 |  | +2.057637 | +2.888437 | +0.939985 | 0.838000 | 76.20 |
| 1.000 | 35.917147 | 0.992711 | 0.992468 | +1.771646 | +3.314757 | +0.052429 | 0.729000 | 144.00 |

The locked curve confirms the main risk-shrinkage shape: higher alpha improves
mean/hard up to medium-high alpha, but the positive ratio and severe tail
deteriorate quickly. WD0375 remains the safer default locked-pass baseline; the
grid is not a license to promote a new locked alpha.

## Reliability

- Haze4K locked test count: `1000` hazy images and `1000` GT images.
- v2.2 parity matched all `1000` images.
- Max absolute parity difference for A0 PSNR/SSIM, WD0375 PSNR/SSIM, and
  WDMamba PSNR/SSIM: `0.0`.

## Evidence

- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616/`
- Primary summary:
  `v210_haze4k_locked_wdmamba_alpha_grid_summary.json`
- Absolute metrics:
  `v210_haze4k_locked_wdmamba_alpha_grid_absolute_metrics.csv`
- Per-image merged table:
  `v210_haze4k_locked_wdmamba_alpha_grid_per_image.csv`
- Parity audit:
  `v210_parity_with_v22_wd0375.json`
