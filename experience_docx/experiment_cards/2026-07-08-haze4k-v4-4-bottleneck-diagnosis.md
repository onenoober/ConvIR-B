# Haze4K v4.4 Bottleneck Diagnosis Route Card

Date: 2026-07-08

Branch: `codex/haze4k-v4-4-bottleneck-diagnosis`

Route id: `haze4k_v4_4_bottleneck_diagnosis_20260708`

Route type: audit-only, no training.

## Fixed v4 Pain Points

1. Spatially non-uniform haze and polluted feature transfer.
2. Low-frequency dehazing versus high-frequency detail preservation conflict.

## Allowed

- Use existing cloud-local A0/A1/A2/A3 checkpoints.
- Diagnose train-derived `trainfit128` and `internal_holdout256`.
- Write compact CSV/JSON/Markdown evidence.

## Forbidden

- Training, checkpoint selection, locked Haze4K test access, A3 extension, density auxiliary, DCFSB on A3, seed sweep, canary expansion.

## Metric Contract

Primary diagnostic split: `internal_holdout256`.

Legacy comparability split: `trainfit128`.

Key output: `joint_delta_matrix.csv` with A1/A2/A3 deltas and A3 interaction delta.

## Current Status

`COMPLETED_NEGATIVE_INTERACTION_CONFIRMED`: v4.4 audit completed. Internal256 prevents an over-negative first128-only reading, but confirms strong non-additive interaction.

Primary internal256 summary:

- A1 mean delta PSNR: `0.060228`
- A2 mean delta PSNR: `0.066864`
- A3 mean delta PSNR: `0.028960`
- expected additive mean: `0.127093`
- A3 interaction mean: `-0.098133`
- A3 positive ratio: `0.484375`

Decision: do not extend A3. Authorize independent v4.5 SDC-Lite and v4.6 DCFSB-bottleneck from the official anchor, with locked test blocked.
