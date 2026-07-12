# v3n Conservative First-Step Calibration

Status: `A0_FAIL_STOP_NO_REPLAY`

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

## A0 Closeout

`v3n_a0_conservative_first_step_closeout.md` records the completed label-only
preflight. R0 failed engineering because the script referenced a misspelled
gate argument; r1 fixed the typo and completed without training, policy replay,
route-confirm, canary, or locked-test access.

The fixed 99th-percentile train-negative threshold equaled
`2.189333099522628e-05` for every operator/fold. With the preregistered strict
`score > threshold` rule, both operators selected zero held-out blocks:
selected coverage `0.0`, positive recall `0.0`, negative false rate `0.0`.

Decision:
`V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY`.

No A1 replay smoke, formal replay, route-confirm, canary, locked test,
training, learned ranker, physics/proxy continuation, or deployment is
authorized from this route.
