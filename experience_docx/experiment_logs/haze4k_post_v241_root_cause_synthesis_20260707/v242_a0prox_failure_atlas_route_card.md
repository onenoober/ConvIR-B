# Haze4K v2.42 A0PROX Failure Atlas Route Card

Route identity: diagnostic-only evidence sync after v2.41 canary32 OOF failure.

Purpose: determine whether v2.41 failed because of residual direction error, scale overshoot, frozen-feature capacity/overfit, or another deployable no-op/unsafe selection failure.

Allowed inputs: v2.41 canary32 OOF compact outputs and Best checkpoints, v2.40 teacher residual alignment atlas where image names overlap, official ConvIR-B A0 checkpoint, and Haze4K train-derived data only.

Forbidden: no training, no parameter update, no canary80, no locked test, no threshold selected for deployment, no new model claim, and no raw images/tensors/checkpoints synced to GitHub.

Required outputs were produced:
- `v242_recompute_audit.json`
- `v242_recompute_mismatch.csv`
- `v242_a0prox_residual_geometry_summary.json`
- `v242_global_shrink_curve.csv`
- `v242_oracle_clamp_upper_bound.json`
- `v242_train_vs_oof_gap.csv`
- `v242_v240_cross_overlap_summary.json`
- `v242_closeout.json`

Result: `A0PROX_DIRECTION_FAIL`.

Key evidence: recompute mismatch `0`; severe `27`; severe direction_bad `27/27`; severe overshoot_bad `0/27`; no shrink gamma passes; oracle mean/hard `+0.0705/+0.1293 dB`; train32 also fails. v2.40 cross-over is coverage-limited to `36/160` OOF images and is supporting context only.
