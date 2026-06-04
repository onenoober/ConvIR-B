# APDR-v0.3 Residual Source Oracle

Date: 2026-06-03

Status: completed full Haze4K diagnostic.

## Scope

This replay-only diagnostic decomposes the v0.2RC safe-mask oracle residual
source. The APDR selector and action mask are frozen. No model weights are
trained or changed.

The variants are:

- `O_full_delta`: `M_safe * clamp(GT - J0, -r, r)`.
- `O_low`: Gaussian low-pass of `Delta_star`.
- `O_high`: `Delta_star - O_low`.
- `O_color`: weighted per-channel affine color correction fitted under `M_safe`.
- `O_low_plus_color`: color affine plus a low-pass residual remainder.

## Command

```bash
cd /root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics
bash experience_docx/experiment_logs/haze4k_apdr_v0_3_residual_source_oracle_20260603/run_apdr_v0_3_residual_source_oracle.sh
```

## Artifacts

- `oracle_residual_source_summary_apdr_v0_3_residual_source_oracle_seed3407.json`
- `oracle_residual_source_per_image_apdr_v0_3_residual_source_oracle_seed3407.csv`
- `oracle_residual_source_apdr_v0_3_residual_source_oracle_seed3407.log`
- `status.txt`

## Results

| Variant | Mean PSNR delta | Hard bottom-25% | Easy top-25% | SSIM delta | Strong regressions | Severe regressions | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `O_full_delta` | `+0.4195` | `+0.8326` | `+0.0085` | `+0.001137` | `0/250` | `0/1000` | pass |
| `O_low` | `+0.3820` | `+0.7893` | `+0.0062` | `+0.000795` | `0/250` | `0/1000` | pass |
| `O_high` | `+0.0296` | `+0.0333` | `+0.0022` | `+0.000346` | `0/250` | `0/1000` | fail |
| `O_color` | `+0.2116` | `+0.5548` | `+0.0010` | `+0.000473` | `0/250` | `0/1000` | pass |
| `O_low_plus_color` | `+0.3788` | `+0.7769` | `+0.0063` | `+0.000794` | `0/250` | `0/1000` | pass |

Mask and residual-source means:

| Statistic | Value |
| --- | ---: |
| `m_safe_mean` | `0.064030` |
| `m_safe_p95_mean` | `0.069366` |
| `delta_star_abs_mean` | `0.015132` |
| `low_delta_abs_mean` | `0.014197` |
| `high_delta_abs_mean` | `0.002928` |
| `color_delta_abs_mean` | `0.012039` |
| `low_plus_color_delta_abs_mean` | `0.014248` |

## Decision

Decision label: `PASS_ORACLE_LOW_COLOR_RESIDUAL_SOURCE`.

The recoverable APDR oracle gain is dominated by low-frequency veil/color
correction rather than high-frequency detail. `O_low` keeps about 91% of the
full mean oracle gain and about 95% of the full hard-bucket oracle gain. The
high-frequency-only source is safe but too weak to justify a detail-first v0.3
route.

The next diagnostic should test whether a deployable low-frequency/color
residual expression can learn this target from hazy/anchor/mask inputs before
any new stop20 training scout.
