# Haze4K v2.24 NoPost Train-Time Controller Failure Audit

Date: 2026-07-04
Status: PLANNED diagnostic-only route

## Purpose

Audit the v2.23 NoPost gated lowband OOF failure as a train-time
controller-transfer failure. This route does not train a new model, does not
touch locked Haze4K test data, and does not select checkpoints or thresholds
from locked test data.

## Source Policy

- Branch: `codex/haze4k-v2-24-nopost-train-time-controller-failure-audit`
- Base: `github/codex/haze4k-official-arch-anchor`
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Reads: v2.23 checkpoints/evidence, v2.21 replay evidence, and earlier
  NoPost oracle evidence where required.

## Declared Phases

- P0 source / evidence / runtime hash audit
- P1 metric comparability audit
- P2 trained risk-head calibration / collapse audit
- P3 post-train factorial action/gate audit
- P4 loss / gradient / supervision audit
- P5 epoch trajectory audit

## Stop Rules

If P2 confirms risk-head collapse or P3 shows action cannot be rescued by
available gates, v2.23 remains paused and locked test remains blocked.
Follow-up must be a new v2.25 diagnostic route, not an expanded v2.23 run.
