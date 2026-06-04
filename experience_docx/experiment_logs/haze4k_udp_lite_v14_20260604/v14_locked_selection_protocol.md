# v1.4 Locked Selection Protocol

Date: 2026-06-04

Status: active policy for ConvIR-Dehaze-v1.4-UDP-Lite.

## Selection Sets

- Use only train-derived internal validation for model/config selection.
- Primary selection split: `val_regular`.
- Hard confirmation split: `val_hard`.
- Training split: `train_inner`.
- Reuse the v1.3 split JSON unless a new split card and split audit supersede it.

## Selection Rule

A checkpoint/config can be considered v1.4A-positive only if Best passes all of:

- `val_regular` mean delta `>= +0.040 dB`.
- `val_regular` easy top-25 delta `>= 0`.
- `val_regular` SSIM delta `>= 0`.
- `val_regular` positive ratio `>= 0.62`.
- `val_regular` strong regression ratio `<= 0.16`.
- `val_regular` worst `<= -0.20 dB` count `<= 12/300`.
- `val_hard` mean delta `>= +0.030 dB`.
- `val_hard` hard bottom-25 delta `>= +0.050 dB`.

Final must also have non-negative `val_regular` and `val_hard` mean deltas.

## Locked Test Rule

- Do not run locked Haze4K test for zero-init, smoke, v1.4A training, scale checks, or module ablations.
- Do not choose checkpoints, active modules, scales, thresholds, or failure subsets using locked Haze4K test.
- If locked test is accidentally used before selection is fixed, relabel that output diagnostic and require a clean fixed-selection rerun before any claim.
- Only one locked Haze4K test is allowed after internal gates pass and the exact checkpoint/config is written.

## Failure Routing

- If v1.4A mean `<= +0.02 dB` and hard `<= +0.02 dB`, stop v1.4A adapter-only and move to v1.4B or external UDPNet reproduction analysis.
- If v1.4A is mean-positive and tail-safe but hard below gate, do not scale-search; move to v1.4B partial unfreeze.
- If v1.4A/B hard and mean pass but tail fails, add AGF-lite only as v1.4C with a new card update.
