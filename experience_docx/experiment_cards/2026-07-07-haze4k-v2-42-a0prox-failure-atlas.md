# Haze4K v2.42 A0PROX Failure Atlas

Route identity: diagnostic-only follow-up to v2.41 canary32 OOF failure.

Purpose: determine whether the failed v2.41 frozen-backbone A0-proximal residual route failed because of residual direction error, scale overshoot, train-vs-OOF overfit/variance, or another deployable selection/no-op identifiability problem.

Allowed inputs: v2.41 canary32 OOF compact outputs and Best checkpoints; v2.40 teacher residual alignment atlas where image names overlap; official ConvIR-B A0 checkpoint; Haze4K train-derived data only.

Forbidden: no training, no parameter update, no canary80, no locked test, no deployment threshold selection, no new model claim, and no raw images/tensors/checkpoints synced to GitHub.

Result: `A0PROX_DIRECTION_FAIL`.

Evidence: tensor-level recompute exactly matched v2.41 (`mismatch_count=0`). The OOF failure reproduced mean/hard/easy `-0.0277/+0.0742/-0.0724 dB`, p05/CVaR5 `-0.3981/-0.5972 dB`, severe `27`, strong-reference regressions `25`, fold pass `0/5`. Residual geometry showed all severe rows were direction failures (`27/27 direction_bad`, `0/27 overshoot_bad`). No global shrink gamma passed; best hard remained `+0.0742 dB` at gamma `1.0`, below the `+0.30 dB` gate and below the stop20 noise scale. Oracle clamp upper bound was weak: positive rate `0.425`, mean/hard/easy `+0.0705/+0.1293/+0.0573 dB`. Train32 full-image evaluation also failed with mean/hard/easy `-0.0239/-0.0722/-0.0206 dB`, severe `26`.

v2.40 cross-over was coverage-limited: only `36/160` v2.41 OOF images appeared in the v2.40 atlas, so it is supporting context rather than a full explanation.

Decision: close the current frozen-backbone small A0-proximal residual family. Do not rescue v2.41 by epochs, folds, sample expansion, loss-weight tuning, beta-only shrink, canary80, locked test, selector/alpha reopening, bridge/generator, or P5. Any future work must be a materially changed GT-risk-controlled architecture/training route, such as A0-anchored partial-unfreeze or a larger non-post ConvIR route with fresh predeclared gates.
