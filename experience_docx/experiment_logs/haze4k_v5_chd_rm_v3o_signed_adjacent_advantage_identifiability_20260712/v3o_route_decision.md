# v3o Route Decision

Start decision:
`V3O_START_A0_SIGNED_ADJACENT_GAIN_TABLE_AND_A1_SCORE_SUFFICIENCY_ONLY`.

Known facts from v3m/v3n: block16 oracle headroom exists, but ordinal
energy-calibration replay is unsafe and the preregistered 99th-percentile
negative energy threshold has zero coverage. Therefore v3o changes the target
from ordinal alpha labels to signed adjacent candidate SSE gain. It does not
reopen either failed policy family.

The only initial authorization is A0 smoke, followed by formal A0 only after
the written smoke integrity gate passes.

## A0 Smoke Decision

`V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`

The 32-image smoke passed for both frozen operators. Fixed `alpha=0.125`
replay was exact to the recorded precision (`0.0 dB` maximum absolute paired
difference), and direct candidate MSE versus summed block SSE differed by at
most `6.073581362234931e-11` for `D_ref` and
`7.561563202841681e-11` for `D_rep`, both below `1e-10`. The source manifest
records `locked_test_touched=false`, `training_occurred=false`, and
`canary_touched=false`.

This validates measurement integrity only. It authorizes exactly formal A0 on
the frozen 1,200-image grouped OOF set; it does not authorize A1, policy
replay, threshold selection, training, canary, route-confirm selection, or
locked-test access.
