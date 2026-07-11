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

## A0b Preflight Decision

`v3m_a0b_metric_contract.md` freezes a no-inference cross-audit. It requires
the v3l 33-level grid and continuous-pixel rows to have the pinned SHA256
values, exact fixed-alpha replay agreement with the v3m five-level rows, and
the same 1,200 OOF names. Its only scientific gate is whether the 95% upper
bound of the dense/continuous advantage over the five-level ladder is at most
`0.005 dB` for every policy/operator pair, with no nested-action monotonicity
violation. No result has been read under this contract yet.
