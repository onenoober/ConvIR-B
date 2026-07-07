# Post-v2.42 Root-Cause Synthesis

Decision: `FROZEN_A0_NEIGHBORHOOD_GT_DESCENT_DIRECTION_FAIL`.

Short label: `FROZEN_A0_GT_DESCENT_FAIL`.

Evidence: v2.42 recomputed v2.41 canary32 OOF exactly with `mismatch_count=0`; severe rows were `27/27 direction_bad` and `0/27 overshoot_bad`; no global shrink gamma passed; best hard delta stayed `+0.0742 dB`; oracle clamp was weak with positive rate `68/160 = 0.425` and mean/hard/easy `+0.0705/+0.1293/+0.0573 dB`; train32 full-image evaluation also failed with mean/hard/easy `-0.0239/-0.0722/-0.0206 dB` and severe `26`.

Interpretation: the blocker is not beta scale, OOF bookkeeping, simple overfit, teacher leakage, or selector absence. The blocker is that the current frozen ConvIR-B plus small A0-proximal residual head does not learn a GT-aligned descent direction around the strong A0 baseline.

Family decision: close the current frozen-backbone small A0-proximal residual family.

Do not reopen v2.41 by more epochs, folds, samples, loss weights, beta-only shrink, canary80, locked test, WDMamba/ConvIR-L alpha continuation, richer target-only selector tuning, M0 bridge/generator, or masked P5 projection.

Allowed next: a materially changed GT-risk-controlled model route, preferably an A0-anchored partial-unfreeze ConvIR route with explicit residual-direction and tail-risk gates.
