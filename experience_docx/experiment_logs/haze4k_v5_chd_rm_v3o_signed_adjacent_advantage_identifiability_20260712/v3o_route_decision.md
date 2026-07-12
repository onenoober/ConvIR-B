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
