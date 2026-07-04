# Haze4K v2.25A NoPost Risk Soft-Label / Scale Distillation Evidence

Status: V225A_RISK_CALIBRATION_GATE_FAIL_NORMAL_PAUSE

Route card:
`experience_docx/experiment_cards/2026-07-04-haze4k-v2-25a-nopost-risk-softlabel-scale-distill.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

Risk-head calibration screen only. Action heads were frozen, and the locked
Haze4K test was untouched.

## Primary Files

- `v225a_preflight.json`: branch, anchor, checkpoint, partial-load, zero-init,
  finite-forward, frozen-action, and locked-test preflight.
- `v225a_oof_summary.json`: OOF calibration metrics and per-fold summaries.
- `v225a_gate.json`: pass/fail gate checks.
- `v225a_closeout.json`: final decision and train/eval provenance.
- `v225a_fold_summary.csv`: compact fold-level metrics.
- `v225a_risk_distill.log`: cloud stdout/stderr capture.
- `status.txt`: launch/done status markers.
- `run_v225a_risk_distill.sh`, `monitor_v225a.sh`: durable command and monitor
  scripts.

Per-fold `*_risk_eval.csv` files and `v225a_oof_risk_eval.csv` remain
cloud-only train-derived diagnostic detail by default. Per-fold `*.pkl`
checkpoints are also cloud-only runtime artifacts and should not be synced to
`main`.

## Summary

- ROC-AUC: `0.5500802065808521`
- AP: `0.4937745923792122`
- ECE10: `0.05257191310326259`
- probability std: `0.0016692509020246116`
- target probability MAE: `0.24880989863237726`
- locked test touched: `False`

## Decision

The calibration gate failed because probability spread, ROC-AUC, and target
probability MAE did not meet the predeclared thresholds. Each fold produced a
constant validation probability (`trained_prob_std=0.0`), confirming that the
risk head remained collapsed under this soft-label / scale-distillation screen.

This is a scientific gate failure, not an infrastructure failure. Per the
predeclared protocol, v2.25A stops here; do not launch post-train factorial
rescue, action joint training, or locked-test evaluation from this result.
