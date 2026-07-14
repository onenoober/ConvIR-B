# v4a Conditional Safety Audit Evidence

Status: `V4A_A0P_COMPLETE_R3_AUTHORIZE_A1F_FEASIBILITY_DESIGN_ONLY`.

This route is an instrumented reconstruction and failure-identification audit
of the closed v3z projected-head contract. It is not a candidate model route,
does not train a new architecture or policy, and cannot access the Haze4K
locked test.

The exact scientific contract is in:

`experience_docx/experiment_cards/2026-07-14-haze4k-v5-chd-rm-v4a-conditional-safety-audit.md`.

The A0R runner imports the immutable v3z source at
`3caddcc5265732e5be77e3404119a28cb28c11e6`, retains raw learned states only
under the cloud run root, and stages only compact manifests, summaries, and
typed closeouts here.

The first A0R launch completed the exact no-op check, then stopped with
`FAILED_COMMAND_OR_INFRA` before projected reconstruction because the wrapper
called a v3p legacy module as if it were the v3w module. That failed output
remains cloud-only under `RUN_ROOT/a0r` and is not scientific evidence.

Corrected run `a0r_r2` then passed the numerical-equivalence gate. Its typed
closeout is `COMPLETED_GATE_PASS` with decision
`V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P`: no-op is exact; historical
versus R1, historical versus R2, and R1 versus R2 aggregate/history maximum
differences are all `0.0`; each reconstruction retains 515 states, 512
projection rows, and 1,280 per-image rows. Canary and locked test remain
untouched.

Committed compact evidence is limited to the typed closeout, reconstruction
summary, and R1/R2 source manifests. Raw states, optimizer/RNG payloads,
per-image tables, projections, renders, and logs remain cloud-only.

After an R3 review, the card now freezes the A0D/A0P state, method, window,
actual-render, bootstrap, and trigger contract. A0D/A0P are not yet implemented
or launched. They remain development-only diagnostics; no result can authorize
A0M, A1, v4b, v4c, canary, or locked test without a new R3 decision.

A0D corrected run `a0d_r2` completed its descriptive gate with 512 finite
image/operator rows and 100 compact group/tail summaries. Its typed closeout is
`COMPLETED_GATE_PASS` with decision
`V4A_A0D_DESCRIPTIVE_COMPLETE_AUTHORIZE_A0P_INTERPRETATION_ONLY`; it selected
no candidate and did not touch canary or locked test. Its raw rows remain at
`RUN_ROOT/a0d_r2/v4a_a0d_a0d_rows_cloud_only.csv`; only the closeout and group
summary are committed. The earlier `a0d_r1` output is a preserved command
failure caused by precreating an output path the frozen source requires to be
new; it is not scientific evidence.

A0P corrected run `a0p_r4` completed all 256 states, all 2,304 factor cells,
both frozen operators, 655,360 raw rows, and the predeclared joint 1,000-draw
bootstrap. Its closeout is structurally valid and records
`A0P_NO_LOCAL_CORRECTION_R3_HANDOFF`. Earlier `a0p_r1` through `a0p_r3` are
command/capacity failures only and contribute no scientific result.

The R3 review found no exact-projection, actual-AdamW-proposal, or risk-window
cell with simultaneous safety improvement. Exact projection was effectively
identical to the historical method; actual-proposal projection had positive
harm/CVaR point effects in every window and both operators while utility
remained within the non-inferiority budget. There was no interaction reversal
and no exact-only positive result.

Decision:
`V4A_A0P_NO_LOCAL_CORRECTION_AUTHORIZE_A1F_METRIC_ALIGNED_FEASIBILITY_ONLY`.
Only a new privileged, v3z-aligned bounded-action feasibility route may be
designed. A0M, optimizer/window retuning, policy replay, candidate training,
canary, and locked test remain blocked. Raw rows and projection diagnostics
remain cloud-only under `RUN_ROOT/a0p_r4`; GitHub receives only the closeout,
step summary, bootstrap summary, window-assignment manifest, and R3 review.
