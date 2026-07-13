# v3w Gradual Safety Ramp Evidence

Status: `COMPLETED_GATE_FAIL`. Compact closeouts, manifests, history, and
summary only; raw cloud outputs remain outside Git.

S0 exact no-op passed for all fixed names and frozen operators. In S1, epochs
1-8 used rendered `.25` MSE only and reached `|Delta u|=0.00134179` with a
`1.12999%` rendered-MSE reduction. Epochs 9-16 linearly ramped the v3s safety
weights from `1/8` to full scale. Final rendered MSE increased from
`0.00032214251` to `0.00032232537` (relative reduction `-0.05676%`), below the
required `0.1%` reduction, while final anchor (`1.1269e-7`), harm
(`2.5852e-7`), and margin (`2.1653e-6`) remained no worse than the fixed v3u
references. Decision: `V3W_S1_RAMP_LOSES_ACTIVITY_STOP_GRADUAL_RAMP`.

This stops the fixed linear safety ramp. The result does not authorize policy,
canary, formal candidate training, deployment, or locked-test access. Do not
continue by testing another fixed safety-weight schedule; a later route needs a
materially different direct low-haze-safety mechanism.
