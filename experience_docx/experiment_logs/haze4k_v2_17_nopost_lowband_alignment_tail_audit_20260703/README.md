# Haze4K v2.17 NoPost Lowband Alignment Tail Audit

Status: `COMPLETED_GATE_FAIL_TRAINING_PAUSED_PENDING_TAIL_AWARE_OBJECTIVE`

Route card:
`experience_docx/experiment_cards/2026-07-03-haze4k-v2-17-nopost-lowband-alignment-tail-audit.md`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Source branch:
`codex/haze4k-v2-17-nopost-lowband-alignment-tail-audit`

Locked Haze4K test: untouched and blocked.

## Plan

Adopt the review recommendation exactly:

> Close WLDB-A as a concrete form; do not close NoPost lowband.

Run no-training audits first:

- R1 WLDB-A postmortem;
- R2 capacity-ladder oracle;
- R3 tail-objective audit only if R2 keeps internal feature lowband open.

No training is launched in R1/R2/R3.

## Evidence Files

Compact GitHub evidence:

- `status.txt`
- `v217_r1_decision.md`
- `v217_r1_closeout.json`
- `v217_r1_wldb_a_checkpoint_pareto.csv`
- `v217_r1_severe_overlap_by_checkpoint.csv`
- `v217_r1_loss_delta_vs_identity_summary.csv`
- `v217_r2_capacity_ladder_protocol.md`
- `v217_r2_decision.md`
- `v217_r2_closeout.json`
- `v217_r2_ladder_summary.csv`
- `v217_r2_group_report.csv`
- `v217_r2_fold_report.csv`
- `v217_r3_decision.md`
- `v217_r3_closeout.json`
- `v217_r3_objective_vs_tail_report.csv`
- `v217_r3_cvar_tail_metrics.csv`
- `v217_r3_budget_activation_report.csv`
- `v217_r3_preserve_mask_report.csv`
- run scripts and compact logs.

Cloud/local raw-output evidence retained but not committed by default:

- per-image R1 loss/action/feature-delta tables;
- per-image R2 O1/O2/O3/O4 oracle tables;
- per-image R3 loss terms;
- v2.16 WLDB-A checkpoints copied from the v2.16 cloud workspace for audit only.

## Result

Cloud run: `convir-4090`, `2026-07-03T17:21:53+08:00` to
`2026-07-03T18:29:09+08:00`.

R1 decision: `R1_CLOSE_WLDB_A_KEEP_NOPOST_LOWBAND_OPEN`.

- WLDB-A `model_5` mean/hard/easy dPSNR:
  `+0.081889/+0.105887/+0.020994`;
- severe count: `67/480`;
- p05 dPSNR: `-0.438669`;
- action-budget term in the v2.16 train history stayed inactive.

R2 decision: `R2_O1_GLOBAL_FEATURE_LL_PASS_REVIEW_WLDB_A2_OBJECTIVE`.

- O1 global final-feature LL oracle passed: mean `+0.842954`,
  hard `+1.591207`, easy `+0.359026`, p05 `+0.001803`, severe rate `0`;
- O2/O3 spatial/internal feature LL oracles also passed with large headroom;
- O4 RGB LL reference remains the upper-bound reference.

R3 decision:
`R3_AVERAGE_OBJECTIVE_IMPROVES_BUT_TAIL_FAILS_REQUIRE_TAIL_AWARE_OBJECTIVE`.

- `model_5` improves average final L1 vs A0 by `-0.000110`;
- but p05/CVaR/severe tail safety fails badly:
  p05 `-0.438669`, CVaR5 `-0.646619`, severe `67/480`;
- validation action-budget activation rate is `0.0`.

## Decision

`NO_TRAINING_PAUSE_DESIGN_TAIL_AWARE_WLDB_A2_OR_WLDB_B_OBJECTIVE`

Close WLDB-A as the concrete trained form. Keep NoPost lowband open because R2
shows strong internal feature-lowband headroom. Do not train the next route
until a materially changed tail-aware objective and gate are written. Locked
Haze4K remains untouched.
