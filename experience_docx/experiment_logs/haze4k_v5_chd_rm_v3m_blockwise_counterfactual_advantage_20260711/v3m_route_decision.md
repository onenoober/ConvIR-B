# v3m Route Decision

Start decision:
`V3M_START_A0A_COMMON_ACTION_GRANULARITY_ONLY_NO_TRAINING_NO_CANARY_NO_LOCKED_TEST`.

v3l established deterministic frozen operators and large oracle headroom, but
block16 and pixel results used different action spaces. A0a isolates that
remaining granularity question with a common ladder. Route-confirm is emitted
only as an audit result and cannot select policies, thresholds, or gates.

Only a dual-operator A0a pass authorizes A0b dense-grid and continuous-pixel
mechanism audits. No other next stage is implied by A0a alone.

The initial frozen replay completed, but its bootstrap-summary implementation
failed before any gate decision. The only authorized recovery is a no-inference
summary rebuild from the existing verified cloud-only raw rows.

## A0a Decision

The constrained rebuild passed the common-action block16 gate for `D_ref` and
`D_rep`. A subsequent compact-only repair changed the operator-agreement reader
from the nonexistent `mean_selected_alpha_mean` field to the raw-table field
`selected_alpha_mean`. It backed up the prior JSON/CSV, read the same raw rows,
and preserved all raw SHA256 values and gate fields exactly.

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

A0b may only cross-audit the already computed dense-grid and continuous-pixel
frozen evidence. It cannot use route-confirm to select a policy and cannot
authorize A1 feasible local actuation, any physics/proxy work, training,
canary, or locked-test access.
