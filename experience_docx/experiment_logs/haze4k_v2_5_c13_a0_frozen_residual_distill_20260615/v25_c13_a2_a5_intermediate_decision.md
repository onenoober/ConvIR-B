# Haze4K v2.5 C13 A2-A5 Intermediate Decision

Date: 2026-06-16

Decision: `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`

## Scope

This decision covers C13-A2 direct-zero residual microfit, A2 post-hoc scale
scan, A3 adaptive scalar microfit, A4 fixed-scale direct microfit, and A5
post-hoc scale sweep on the A4 checkpoint.

All runs used train-derived C13 split/evidence only on `convir-4090`. Locked
Haze4K test remained untouched, and locked per-image outputs were not read.

## Key Results

| Variant | val count | mean | hard | easy | positive | severe / 600 | dSSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 direct-zero 256 scale 1.00 | 128 | `+0.310894` | `+0.598608` | `+0.070371` | `0.695312` | `131.25` | `+0.00016295` |
| A2 post-hoc scale 0.25 | 128 | `+0.142072` | `+0.153970` | `+0.151167` | `0.828125` | `14.0625` | `+0.00008881` |
| A2 post-hoc scale 0.50 | 128 | `+0.220755` | `+0.304979` | `+0.163641` | `0.781250` | `60.9375` | `+0.00012478` |
| A3 adaptive 0.25 | 128 | `+0.056464` | `+0.017318` | `+0.110704` | `0.890625` | `0.0` | `+0.00005316` |
| A3 adaptive 0.50 | 128 | `+0.064695` | `+0.025806` | `+0.119304` | `0.843750` | `0.0` | `+0.00005749` |
| A4 fixed scale 0.50 | 128 | `+0.317922` | `+0.604817` | `+0.088566` | `0.718750` | `131.25` | `+0.00016756` |
| A4 fixed scale 0.55 | 128 | `+0.317983` | `+0.606338` | `+0.086847` | `0.718750` | `131.25` | `+0.00016672` |
| A5 A4 sweep scale 0.25 | 128 | `+0.221040` | `+0.307825` | `+0.163525` | `0.796875` | `51.5625` | `+0.00012394` |
| A5 A4 sweep scale 0.30 | 128 | `+0.247162` | `+0.368207` | `+0.159603` | `0.781250` | `75.0` | `+0.00013577` |
| A5 A4 sweep scale 0.35 | 128 | `+0.269935` | `+0.428158` | `+0.150080` | `0.781250` | `84.375` | `+0.00014607` |
| A5 A4 sweep scale 0.40 | 128 | `+0.289329` | `+0.487620` | `+0.135008` | `0.757812` | `98.4375` | `+0.00015481` |
| A5 A4 sweep scale 0.45 | 128 | `+0.305326` | `+0.546529` | `+0.114467` | `0.734375` | `121.875` | `+0.00016198` |

Quick gate:

```text
mean >= +0.25
hard >= +0.35
easy >= +0.10
positive >= 0.80
severe <= 60/600
dSSIM >= 0
```

No A2-A5 row passed the quick gate.

## Interpretation

C13 confirmed that A0-frozen residual learning is learnable and can produce
real positive movement, unlike C12 direct full-model distillation. However, the
current residual adapter still exposes a hard tradeoff:

```text
small residual scale -> safe positive/tail, but mean/hard too small
large residual scale -> strong mean/hard, but positive and severe tail fail
adaptive scalar -> too conservative, hard gain collapses
```

The best safety-leaning row is A5 scale `0.25`, but it misses mean, hard, and
positive slightly. The best hard-gain rows are A4/A2 high-scale variants, but
their severe tail is far beyond the quick gate.

## Decision

Do not start C13-B adapter screen from the current C13 residual adapter/loss.
Do not run full 600 validation or formal 5x3 for these variants. Do not touch
locked Haze4K.

Recommended next route, if C13 is reopened, should add explicit risk/utility
conditioning or a per-image/per-region no-op gate before any larger screen.
Continuing only LR, epoch, or global residual-scale tuning is not justified by
the A2-A5 evidence.
