# Haze4K CHD-RM v3i FAM2 Open-Value Distillability

Date: 2026-07-11
Branch: `codex/haze4k-v5-v3i-fam2-open-value-distillability`
Evidence:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711/`

## Purpose

v3g proved that FAM2 action-space oracle selection is strong. v3h showed that
the audited scalar/operator-site features cannot recover that oracle. v3i tests
whether the open-value target itself is compressible into a controller-realistic
spatial policy before any controller training is allowed.

## Stage Gate

v3i-A runs no training. It audits `open_score = -dL/dalpha at alpha=0` inside
D7c active sites and replays compressed oracle policies. v3i-B is authorized
only if a compressed policy retains at least 25% of the gap from hard D7c to
open top-50 oracle without exceeding hard D7c severe regressions.

## Locked Test Policy

Locked Haze4K test remains sealed. This route uses only internal train-derived
`val_inner` evidence unless a later written gate explicitly changes that.
