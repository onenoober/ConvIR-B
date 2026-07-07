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

`PLANNED`: audit script pending.
