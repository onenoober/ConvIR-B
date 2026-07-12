# v3n Route Decision

Start decision:
`V3N_START_A0_CONSERVATIVE_FIRST_STEP_LABEL_PREFLIGHT_ONLY`.

v3n is a new target-semantics diagnostic after v3m A3 fail-stop. It is not a
rescue of the failed v3m calibrated policy. A0 may only run a label-only
preflight using the fixed conservative rule in `v3n_a0_metric_contract.md`.

No training, route-confirm, canary, locked-test access, policy replay,
threshold-family search, physics/proxy continuation, or deployment is
authorized at route start.

## A0 Decision

The r1 label-only preflight applied the fixed contract to the v3m-A1 block
table. For every operator/fold, the 99th percentile of train-fold negative
`direct_step_energy` was `2.189333099522628e-05`. With the preregistered strict
`score > threshold` rule, selected coverage and positive recall were `0.0` for
both frozen operators.

Decision:
`V3N_A0_CONSERVATIVE_LABEL_PREFLIGHT_FAIL_STOP_NO_REPLAY`.

No A1 replay smoke, formal replay, route-confirm, canary, locked-test access,
training, learned ranker, physics/proxy continuation, or deployment is
authorized.
