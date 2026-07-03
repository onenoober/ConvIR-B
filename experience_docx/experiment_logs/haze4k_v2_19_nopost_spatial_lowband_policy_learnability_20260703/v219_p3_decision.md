# v2.19 P3 Tail-Aware Objective Replay For Spatial Predictions

Decision: `P3_PASS_SPATIAL_OBJECTIVE_REPLAY_COVERS_FAILURES`

- primary variant: `P1_small_cnn_spatial`
- severe coverage by tail hinge: `1.0`
- positive tail-hinge activation rate: `0.0`
- strong/easy regression coverage by preserve hinge: `1.0`
- positive preserve-hinge activation rate: `0.0`
- oracle_p75 budget activation rate: `0.0975`
- oracle_p75 safe-oracle over-penalized rate: `0.25`

P3 is diagnostic/replay evidence only; it does not train WLDB-B.
