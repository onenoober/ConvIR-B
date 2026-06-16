# Haze4K v2.8b NH-HAZE Official-Test Split Audit

Status: `COMPLETED_AUDIT_SPLIT_LEAKAGE_FOUND`

- Source route: `experience_docx/experiment_logs/haze4k_v2_8_nhhaze_official_weights_20260616/`
- Source per-image table: `v28_nhhaze_official_weights_per_image.csv`
- Route card: `experience_docx/experiment_cards/2026-06-16-haze4k-v2-8-nhhaze-official-weights.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Decision: `V28B_NHHAZE_OFFICIAL_TEST51_55_REPRO_ALIGNED_WITH_EXPECTED_BASELINES`
- New model runtime: `false`
- Created from existing v2.8 per-image metrics only: `true`
- Haze4K locked test touched: `false`

## Audit Finding

The original v2.8 run used the NH-HAZE-specific ConvIR-B and WDMamba
checkpoints, but it evaluated the flat local directory
`/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/` as all `55` paired images.
That is not a valid official NH-HAZE benchmark split.

Official-style NH-HAZE evaluation must separate the `55` images as:

```text
01-45: train
46-50: validation
51-55: test
```

The ConvIR README also reports NH-HAZE ConvIR-B at `20.66/0.802`, and the
WDMamba checkpoint name is `NH_20.83.pth`. Re-aggregating only `51-55` aligns
with both expected baselines:

```text
A0_NH / ConvIR-B on 51-55: 20.6636 PSNR / 0.7968 SSIM
WDMamba_NH on 51-55:      20.8307 PSNR / 0.8182 SSIM
```

The inflated all-55 A0 result `26.1047/0.9296` was caused by mixing official
train/validation images into the benchmark aggregation.

## Split Metrics

| Split | Role | Count | A0 PSNR/SSIM | WDMamba PSNR/SSIM |
| --- | --- | ---: | ---: | ---: |
| `01-55` | mixed train/val/test, invalid for official benchmark | 55 | `26.1047/0.9296` | `23.9070/0.8684` |
| `01-45` | official train | 45 | `26.7379/0.9444` | `24.3867/0.8781` |
| `46-50` | official validation | 5 | `25.8473/0.9292` | `22.6659/0.8314` |
| `51-55` | official test | 5 | `20.6636/0.7968` | `20.8307/0.8182` |

## Official-Test Alpha Diagnostic

On the official `51-55` test subset, `alpha=0.375` is positive, but this is
still a post-run diagnostic observation on only five images and must not be
used as a selected NH-HAZE alpha without a separate validation/OOF protocol.

| Alpha | Mean dPSNR | Mean dSSIM | Positive | Severe | Worst |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.125` | `+0.224743` | `+0.00889719` | `1.0000` | `0/5` | `+0.100395` |
| `0.250` | `+0.398761` | `+0.01628566` | `1.0000` | `0/5` | `+0.127104` |
| `0.375` | `+0.515796` | `+0.02203439` | `1.0000` | `0/5` | `+0.078772` |
| `0.500` | `+0.571267` | `+0.02600487` | `0.8000` | `0/5` | `-0.042160` |
| `0.750` | `+0.490403` | `+0.02805877` | `0.6000` | `1/5` | `-0.475842` |
| `1.000` | `+0.167149` | `+0.02141026` | `0.6000` | `2/5` | `-1.103455` |

## Primary Files

- `v28b_nhhaze_official_test_split_audit_summary.json`
- `v28b_nhhaze_split_absolute_metrics.csv`
- `v28b_nhhaze_split_alpha_grid.csv`
- `v28b_nhhaze_official_test_51_55_per_image.csv`
- `v28b_decision.md`
