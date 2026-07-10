# v3i Route Decision

Route: `haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711`
Branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
Date: 2026-07-11

## Route Identity

This is a new diagnostic/no-training audit route after v3h. It is not a v3d
continuation, not v3f-B ranker training, not a router canary, and not a model
structure route.

## Source Of Truth

- GitHub `main` commit `b267d9f`: v3h evidence sync and current CHD-RM state.
- Cloud runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3i-fam2-open-value-distillability`.
- Parent runnable code: v3h operator-site context audit branch at `0d0d79e`.

## Authorized Stage

Only v3i-A is authorized initially: no-training FAM2 open-value teacher
compressibility audit on internal train-derived `val_inner` 600.

## Forbidden Flow

- No v3d continuation.
- No v3f-B scalar ranker training.
- No controller/router training in v3i-A.
- No canary expansion.
- No 20-epoch continuation.
- No v4/RARM expansion.
- No backbone/FAM1/neighbor unfreeze.
- No locked Haze4K test.
- No checkpoints, weights, raw arrays, or images in GitHub evidence.
