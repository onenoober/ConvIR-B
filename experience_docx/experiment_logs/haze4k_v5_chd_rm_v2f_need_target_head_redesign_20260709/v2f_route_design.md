# v2f Route Design

v2e proved that D7c top-k learns a real density-beyond-baseline `R_need` signal,
but no D7c/RP operating point satisfies LDHN recall and false-tail safety
together. v2f therefore changes the question from "increase recall penalty" to
"is the target/head separable inside density strata?"

The first stage is diagnostic:

- F0 reproduces the v2e source-of-truth metrics and records no locked-test use.
- F1 audits LDHN stability, boundary sensitivity, isolated support, and
  adjacency to denser haze.
- F2-lite probes whether frozen A0/D3/multi-context features separate LDHN from
  low-density low-need hard negatives.
- F3 compares global, density-conditioned, and excess-over-density need targets.

F4 density-stratified head training is held until F1/F2/F3 justify it.

F1/F2/F3 now justify a small F4 canary because LDHN is not dominated by
unstable pixels, frozen features separate LDHN from low-density low-need
negatives, and density-conditioned targets remove the global density proxy
signal. F4 remains a frozen-side head canary only; it does not authorize D2,
v3, RARM, or locked Haze4K test.
