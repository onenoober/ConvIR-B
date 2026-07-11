# v3j Route Decision

Route: `haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711`
Branch: `codex/haze4k-v5-v3j-bounded-safe-correction-audit`
Date: 2026-07-11
Status: `PLANNED`

## Route Identity

This is a new diagnostic audit after v3i. It is not a v3d continuation, not
v3f-B ranker training, not a larger FAM2 router, and not a model-structure
canary.

## Source Of Truth

- GitHub `main`: v3i closeout and FAM family summary.
- Parent runnable branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
  at `8517614`.
- Cloud workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3j-bounded-safe-correction-audit`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3j_bounded_safe_correction_audit_20260711/`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Question

Can a privileged GT-derived safe teacher supervise a bounded output residual
correction action space, instead of supervising a post-hoc switch for the old
FAM2 correction?

## Stage Gates

v3j-A runs first and performs no training. It calibrates per-channel residual
bounds on fresh train-derived `v3j_controller_calib` and evaluates bounded
teacher projections on fresh train-derived `v3j_route_confirm`.

v3j-B is authorized only if a primary-teacher bounded projection beats hard D7c
with positive bootstrap lower bound, non-worse p10, and non-worse
severe-regression count.

## Forbidden Flow

No locked Haze4K test, no v3d continuation, no v3f-B ranker, no current-signal
FAM2 router, no backbone/adapter/FAM producer training, no canary expansion
before v3j-B pass, and no changing teacher/bound/gate after seeing
route-confirm metrics.
