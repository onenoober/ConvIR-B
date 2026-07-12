# v3n Route Decision

Start decision:
`V3N_START_A0_CONSERVATIVE_FIRST_STEP_LABEL_PREFLIGHT_ONLY`.

v3n is a new target-semantics diagnostic after v3m A3 fail-stop. It is not a
rescue of the failed v3m calibrated policy. A0 may only run a label-only
preflight using the fixed conservative rule in `v3n_a0_metric_contract.md`.

No training, route-confirm, canary, locked-test access, policy replay,
threshold-family search, physics/proxy continuation, or deployment is
authorized at route start.
