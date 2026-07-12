# v3o Action Contract

The fixed candidate ladder is exactly `[0.0, 0.125, 0.25, 0.5, 1.0]` applied
to the frozen direct correction with the same per-channel clamp used in v3m.

During A0/A1 these are counterfactual measurement candidates only. No policy is
executed. Any future deployable route defaults to `0.125` and may consider only
the adjacent `0.125 -> 0.25` transition until a separate written gate authorizes
another transition.
