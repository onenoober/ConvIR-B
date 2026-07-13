# v4a Conditional Safety Audit Evidence

Status: `V4A_A0R_PASS_A0DP_AMENDMENT_FROZEN_AWAITING_R2_IMPLEMENTATION`.

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
