# Haze4K DPGA Tail-Control Route Summary

Date: 2026-06-04

Status: completed diagnostic route; no locked Haze4K test allowed.

## Route

The DPGA route tested whether in-network depth/prior-guided adapters could
increase Haze4K hard-case PSNR while controlling strong-reference, easy-image,
and worst-tail regressions. Training and evaluation ran only on AutoDL
`autodl-dehaze4`; local work was limited to code and static checks.

## Runtime Diagnostics

The DPGA-Lite runtime diagnostics evaluated module ablation and effective scale
sweeps on existing Best/Final checkpoints. The decision selected shallow-only
DPGA with reduced scale before new training, because this was the safer
configuration for mean movement and tail risk.

## v1.1

Model:
`ConvIR-Haze4K-DPGA-v1.1-tail-control-shallow-scale0p25-seed3407-20260604`

Config: shallow-only, scale `0.25`, anchor `0.08`, FFT `0.05`, chroma `0.03`,
delta `0.0002`, TV `0.00005`.

Gate result: fail.

| Check | Observed | Required | Result |
| --- | ---: | --- | --- |
| Best mean PSNR delta | `+0.037036 dB` | `>= +0.030 dB` | pass |
| Best hard bottom-25% delta | `+0.023367 dB` | `>= +0.030 dB` | fail |
| Best easy top-25% delta | `+0.058178 dB` | `>= 0` | pass |
| Best positive ratio | `0.623333` | `>= 0.55` | pass |
| Best strong regression ratio | `0.16` | `<= 0.42` | pass |
| Best worst `<= -0.20 dB` | `9/300` | `<= 36/300` | pass |

Interpretation: v1.1 was not a collapse, but hard bottom-25% gain was below
the written gate. Locked Haze4K test stayed blocked.

## v1.2

Model:
`ConvIR-Haze4K-DPGA-v1.2-hard-gain-shallow-scale0p5-anchor0p04-seed3407-20260604`

Config: shallow-only, scale `0.5`, anchor `0.04`, FFT `0.05`, chroma `0.03`,
delta `0.00025`, TV `0.00005`.

Gate result: fail.

| Check | Observed | Required | Result |
| --- | ---: | --- | --- |
| Best mean PSNR delta | `+0.042656 dB` | `>= +0.030 dB` | pass |
| Best hard bottom-25% delta | `+0.026225 dB` | `>= +0.030 dB` | fail |
| Best easy top-25% delta | `+0.068009 dB` | `>= 0` | pass |
| Best positive ratio | `0.62` | `>= 0.55` | pass |
| Best strong regression ratio | `0.186667` | `<= 0.42` | pass |
| Best worst `<= -0.20 dB` | `16/300` | `<= 36/300` | pass |

Interpretation: v1.2 increased mean/easy gain but still missed hard bottom-25%.
It also raised worst-tail regressions relative to v1.1. The failure analysis
therefore says not to launch a higher-scale follow-up without a new diagnostic.

## Decision

`STOP_DPGA_SCALE_ONLY_TAIL_CONTROL`.

Do not run locked Haze4K test for v1.1 or v1.2. Future DPGA work needs a new
mechanism or preflight that specifically raises hard bottom-25% gain without
increasing worst-tail regressions.
