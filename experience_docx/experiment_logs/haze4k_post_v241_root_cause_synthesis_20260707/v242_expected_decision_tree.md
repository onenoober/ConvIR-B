# v2.42 Expected Decision Tree

Allowed labels:
- `A0PROX_DIRECTION_FAIL`
- `A0PROX_SCALE_FAIL_BUT_DIRECTION_EXISTS`
- `A0PROX_SELECTION_FAIL_AGAIN`
- `A0PROX_OVERFIT_VARIANCE_FAIL`

Actual v2.42 branch: `A0PROX_DIRECTION_FAIL`.

Why not scale-only: severe rows are not overshoot-dominant (`0/27` overshoot_bad), no global shrink gamma passes the gate, and shrinking removes hard gain rather than revealing a deployable scale.

Why not selector again: the oracle clamp upper bound is weak, with mean/hard only `+0.0705/+0.1293 dB`; it does not provide enough positive action value to justify reopening selector work.

Why not overfit/variance as the main label: train32 full-image evaluation also fails, with mean/hard/easy `-0.0239/-0.0722/-0.0206 dB`, severe `26`, and gate pass false.

Decision consequence: close the current frozen-backbone small A0-proximal residual family. Only a materially changed capacity/representation route may be opened next.
