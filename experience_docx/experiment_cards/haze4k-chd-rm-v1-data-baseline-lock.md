# Haze4K v5 CHD-RM v1 Data Baseline Lock

Date: 2026-07-08

Status: completed; decision `COMPLETED_V1_GATE_PASS`.

## Scope

- Route: CHD-RM v5.
- Stage: v1 data and ConvIR-B baseline lock.
- Branch: `codex/haze4k-v5-v1-chd-rm-data-baseline-lock`.
- Source code anchor: `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` via v0 docs-only branch.
- Runtime: `convir-4090` only.
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Baseline checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Locked test: not used for tuning; v1 may only count files and verify path integrity.

## v1 Gates

- train image pairs = 3000 after image-extension filtering.
- test image pairs = 1000 after image-extension filtering.
- train/test hash leakage = none.
- internal split = 2400 train / 600 val.
- OOF folds = 5 folds of 600 validation images each.
- A0 val600 metrics and metric reproducibility must be completed before v2.

## Current Decision

Data manifest, leakage audit, A0 val600, metric reproducibility, and efficiency evidence completed. Decision: `COMPLETED_V1_GATE_PASS`.
