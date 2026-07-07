# Haze4K v2.42 A0PROX Failure Atlas

Status: completed diagnostic-only on `convir-4090` at 2026-07-07 11:45 +08.

Decision: `A0PROX_DIRECTION_FAIL`.

## Source Gate

Authoritative status came from GitHub `main` experiment evidence and cloud runtime assets. This route uses the completed v2.41 canary32 OOF outputs and checkpoints plus the v2.40 teacher residual alignment atlas where image names overlap.

Cloud inputs:
- v2.41 evidence: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual/experience_docx/experiment_logs/haze4k_v2_41_a0_proximal_supervised_residual_20260706/`
- v2.41 fold Best checkpoints: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual/Dehazing/ITS/results/ConvIR-Haze4K-v2.41-a0prox-canary32-oof-20260706/fold_*/Best.pkl`
- v2.40 atlas CSV: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-40-teacher-residual-alignment-atlas/experience_docx/experiment_logs/haze4k_v2_40_teacher_residual_alignment_atlas_20260706/v240_teacher_residual_alignment_atlas_per_image.csv`
- data split: Haze4K train-derived only.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Forbidden actions remained blocked: no training, no parameter update, no canary80, no locked test, no deployable threshold selection, and no new model claim.

## Metric Contract

The route reuses the v2.41 gate contract: global and fold OOF metrics compare A0 vs A0PROX on the same held-out images; hard/easy are bottom/top 25% by A0 PSNR; severe is `delta_psnr <= -0.20`; strong-reference regression is easy-bucket `delta_psnr < -0.01`; promotion-style gate requires mean `>= +0.15`, hard `>= +0.30`, easy `>= 0`, p05 `>= -0.01`, CVaR5 `>= -0.02`, no severe, no strong-reference regressions, and at least 4/5 fold pass.

## Results

Recompute audit passed exactly: `v242_recompute_audit.json` reports `mismatch_count=0`, `missing_count=0`, and all tracked max absolute diffs are `0.0`. This confirms the v2.41 severe/worst rows were not a bookkeeping mismatch against the stored canary32 OOF table.

OOF metrics reproduced v2.41: mean/hard/easy `-0.027738/+0.074246/-0.072387 dB`, p05/CVaR5 `-0.398078/-0.597185 dB`, worst `-1.266457 dB`, severe `27`, strong-reference regressions `25`, fold pass `0/5`.

Residual geometry points to direction failure, not scale overshoot:
- severe rows: `27/27` are `direction_bad` (`alignment_dot >= 0`).
- severe rows: `0/27` are `overshoot_bad`.
- alpha safe upper p05/median: `-20.258853/-0.949031`.
- residual identity check max absolute error: `9.60e-11`.

Global shrink curve did not rescue the route. No gamma passed the gate. The best hard gain is still at gamma `1.0` with only `+0.074246 dB`, below the `+0.30 dB` hard gate and below the known stop20 noise scale. Tiny gamma values reduce tail damage but also remove useful hard gain; gamma `0.03125` has hard `+0.002503 dB` and still has p05 `-0.012503` plus 4 strong-reference regressions.

Oracle clamp upper bound is weak: positive rate `68/160 = 0.425`, oracle mean/hard/easy `+0.070497/+0.129325/+0.057274 dB`, p05/CVaR5 `0.0/0.0`, severe `0`. This is below the mean/hard gate and does not justify a new selector route.

Train-vs-OOF does not support a simple overfit explanation. Full-image train32 evaluation also fails: mean/hard/easy `-0.023893/-0.072160/-0.020582 dB`, p05/CVaR5 `-0.490251/-0.682147 dB`, severe `26`, strong-reference regressions `17`. OOF32 similarly fails with severe `27`.

v2.40 cross-over is coverage-limited: only `36/160` v2.41 OOF images appear in the v2.40 atlas (`coverage_fraction=0.225`). In that overlap, only 5 severe v2.41 images are joinable; 0 overlap WDMamba unsafe, 2 overlap ConvIR-L unsafe. Treat this as supporting context only, not a full v2.41 failure explanation.

## Closeout

The v2.42 diagnostic closes the frozen-backbone small A0-proximal residual family as currently formulated. v2.41 must not be continued by epochs, folds, sample expansion, loss-weight tuning, beta-only shrink, canary80, or locked test. The next model route, if opened, must be materially changed, such as an A0-anchored partial-unfreeze or larger GT-risk-controlled ConvIR route with its own predeclared gates.

Compact outputs:
- `v242_recompute_audit.json`
- `v242_recompute_mismatch.csv`
- `v242_a0prox_residual_geometry_summary.json`
- `v242_global_shrink_curve.csv`
- `v242_oracle_clamp_upper_bound.json`
- `v242_train_vs_oof_gap.csv`
- `v242_v240_cross_overlap_summary.json`
- `v242_closeout.json`

Cloud-only by default: `v242_a0prox_residual_geometry_per_image.csv` and runtime logs.
