# v3n Conservative First-Step Calibration

Status: `PLANNED_A0_LABEL_ONLY_PREFLIGHT`

Route card:
`experience_docx/experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3n-conservative-first-step-calibration.md`

v3n tests a materially different target semantics from v3m A2/A3: a
false-intervention-protected, label-only, first-step rule. It defaults to
`alpha=0.125` and permits only `alpha=0.25` on blocks whose
`direct_step_energy` exceeds the fixed 99th percentile of train-fold negative
blocks.

No training, route-confirm, canary, locked test, policy replay, threshold-family
search, or physics/proxy continuation is authorized in A0.

## A0 Contract

See `v3n_a0_metric_contract.md`.
