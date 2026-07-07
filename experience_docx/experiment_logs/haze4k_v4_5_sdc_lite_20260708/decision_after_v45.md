# Haze4K v4.5 SDC-Lite Decision

Status: stage gate failed; route not promoted.

Locked test: not touched; not enumerated.

## Metrics

| Split | Mean dPSNR | Positive ratio | p5 dPSNR | Mean dSSIM | R std mean | R corr input L1 | R corr A0 L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trainfit128 | -0.014244 | 0.484375 | -0.195040 | -0.00010642 | 0.078158 | -0.532929 | -0.505587 |
| internal256 | -0.009711 | 0.437500 | -0.180751 | -0.00002281 | 0.082352 | -0.467570 | -0.409864 |

## Decision

Do not expand v4.5 immediately. Internal256 is negative, positive ratio is below threshold, `R_1_2_std_mean` is below threshold, and the learned response is negatively correlated with both haze and A0-error proxies. This is enough evidence to avoid a short-run false positive, but not enough to abandon the two v4 pain points.

Next authorized phase: independent v4.6 DCFSB-bottleneck route from `codex/haze4k-official-arch-anchor`.

Raw cloud-only files not committed by default: `v45_sdc_lite_per_image.csv`, `v45_sdc_lite_module_stats.jsonl`, and checkpoint files under `Dehazing/ITS/results/`.
