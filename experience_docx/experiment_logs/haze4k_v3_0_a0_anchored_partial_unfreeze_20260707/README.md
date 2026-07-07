# Haze4K v3.0 A0-Anchored Partial-Unfreeze

Status: completed canary32 OOF on `convir-4090`.

Decision: `V300_CANARY32_FAIL_TAIL_DIRECTION_RISK_LOCK_CANARY80_LOCKED_TEST`.

## Source And Contract

This route starts from immutable `github/codex/haze4k-official-arch-anchor` and tests a materially changed model-line route after v2.42 closed the frozen small A0PROX family. It uses Haze4K train-derived data only and reuses the v2.41 canary32 OOF split for direct comparison.

Forbidden actions remained blocked throughout: no canary80, no locked test, no full unfreeze, no teacher target as primary objective, and no raw image/tensor/checkpoint sync to GitHub.

Metric gates: mean `>= +0.15 dB`, hard `>= +0.30 dB`, easy `>= 0`, p05 `>= -0.01`, CVaR5 `>= -0.02`, severe `0`, strong-reference regressions `0`, fold pass `>= 4/5`, and severe_direction_bad `0`.

## Stage-0

Stage-0 passed after one engineering-invalid train-mode preflight attempt was fixed by evaluating the wrapper in eval mode. The valid preflight reports:
- identity max abs vs official A0: `0.0`
- finite outputs: true
- forbidden symbol hits: `0`
- locked test touched: false
- trainable parameter counts: frozen_probe `11875`, tier_a_partial `212662`, tier_b_partial `1148742`

## Canary32 Results

`frozen_probe` trained only `V300_*` and failed:
- mean/hard/easy `-0.0008/+0.0146/+0.0089 dB`
- p05/CVaR5 `-0.1387/-0.2109 dB`
- severe `3`, strong-reference regressions `18`, fold pass `0/5`
- severe direction_bad `3/3`

`tier_a_partial` trained `V300_*`, `Decoder.2`, `Convs.1`, and `feat_extract.5` and failed:
- mean/hard/easy `+0.0072/+0.0718/+0.0197 dB`
- p05/CVaR5 `-0.3751/-0.5918 dB`
- severe `22`, strong-reference regressions `18`, fold pass `0/5`
- severe direction_bad `22/22`

`tier_b_partial` trained `V300_*`, `Decoder.1/2`, `Convs.0/1`, and `feat_extract.3/4/5` and failed:
- mean/hard/easy `+0.0050/+0.2226/-0.0889 dB`
- p05/CVaR5 `-0.6924/-1.0190 dB`
- severe `42`, strong-reference regressions `19`, fold pass `0/5`
- severe direction_bad `40/42`, severe overshoot_bad `2/42`
- oracle upper bound was near but still not gate-clean: mean/hard/easy `+0.1533/+0.2962/+0.1159 dB`

## Interpretation

The materially changed v3.0 route did move beyond the v2.41 small frozen residual setup, but canary32 still failed. The frozen probe stayed too weak. Tier-A did not rescue residual direction. Tier-B produced larger hard movement but converted it into severe tail/easy damage and mostly direction_bad failures. This does not authorize canary80, locked test, selector reopening, or simple wider decoder tuning.

Compact files in this directory are GitHub-syncable. Per-image CSVs and checkpoints remain cloud-only by default.
