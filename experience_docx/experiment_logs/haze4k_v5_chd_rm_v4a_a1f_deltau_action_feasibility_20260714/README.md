# v4a-A1F Delta-u Action Feasibility Evidence

Status: `PLANNED` after an engineering-only S0 repair.

This route is a privileged, train-derived feasibility oracle after v4a A0P
found no local optimizer/projection/window correction. It restores the exact
A0R r1 final v3z head and tests whether a bounded direction-line action set
adds safe heldout128 headroom beyond privileged shrink/abstention.

The canonical contract is:

`experience_docx/experiment_cards/2026-07-14-haze4k-v5-v4a-a1f-deltau-action-feasibility.md`.

No training, policy fitting, canary, candidate selection, or locked-test access
is permitted. Raw per-image/action rows remain under cloud `RUN_ROOT`; only the
source manifest, typed closeout, operator summary, bootstrap summary, and this
README may be compact GitHub evidence.

The first smoke attempt, `v4a_a1f_s0_smoke_r1` at route commit `ff9cf921`,
stopped before a gate decision after 8 update and 3 heldout images. The fixed
shrink grid re-reduced its guaranteed zero action in a batched float32 kernel;
the `1e-12` comparison could then exclude the exact predecessor. The repair
requires bitwise equality of the zero-action rendered tensors and canonicalizes
only that identical candidate to the already replayed predecessor metrics.
Other candidates and all scientific thresholds are unchanged. See
`v4a_a1f_smoke_r1_failure_closeout.json`.
