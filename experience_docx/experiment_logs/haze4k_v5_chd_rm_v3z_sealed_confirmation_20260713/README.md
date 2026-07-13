# v3z Sealed Confirmation Evidence

Status: `COMPLETED_GATE_FAIL`. Only compact manifests, closeouts, history, and
summaries belong here. Raw cloud outputs remain outside Git.

S0 passed exact no-op over train128. S1 is `COMPLETED_GATE_FAIL`: train128
retained activity and a `4.14653%` rendered-MSE reduction, but heldout128's
anchor and harm exceeded the fixed v3u references despite `1.35612%` rendered
MSE reduction. Decision: `V3Z_S1_SEALED_CONFIRMATION_FAIL_CLOSE_PROJECTED_HEAD_ROUTE`.
No further projected-head tuning, policy, canary, candidate training, or locked
test is authorized.
