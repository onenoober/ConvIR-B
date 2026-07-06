# Haze4K v2.38 Micro-Alpha Same-Context WDMamba Safe Substrate and Projection Audit

Date: 2026-07-06

Branch:
`codex/haze4k-v2-38-microalpha-same-context-wdmamba-safe-substrate-projection`

Route identity: new conservative same-context teacher-contract audit. This is
not a continuation of v2.37 masked/oracle bridge work.

Current bottleneck:
`TARGET_ONLY_NOOP_UNSAFE_SIGNAL_NOT_DEPLOYABLE`.

## Primary Question

Can a smaller global same-context WDMamba alpha produce a fold-stable,
tail-safe, unmasked teacher substrate that does not require runtime no-op/unsafe
selection?

## Inherited Facts

- v2.37 M0 oracle-positive plus A0-preservation passed P2/P3 but is not
  deployable because target-only unsafe/no-op separability failed P4.
- v2.37 alpha0.125 had positive mean/hard/easy and severe `0`, but still had
  `3` strong-reference regressions and fold pass `2/5`.
- v2.35/v2.36 established that the valid WDMamba view is full-image
  same-context, not direct 256-crop WDMamba and not crop-input/fullslice
  student training.

## Not Allowed

- No oracle mask bridge training.
- No generator training.
- No v2.37 P5 masked free-tensor projection.
- No canary80.
- No locked test.
- No direct WDMamba-on-256-crop teacher.
- No 256 crop-input/full-image-slice target.
- No S5-only continuation.
- No relaxation of strong-reference, severe, or fold gates.

## Metric Contract

P0 uses cached full-image A0 and WDMamba tensors already audited by v2.35/v2.37.
It recomputes alpha blends offline against Haze4K train-derived GT and does not
rerun WDMamba. The alpha grid is:

`0.015625, 0.03125, 0.046875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125`.

Buckets reuse the v2.37 full-image A0 PSNR definition: bottom 25% is hard, top
25% is easy, and strong-reference is A0 PSNR greater than or equal to the 75th
percentile. Fold IDs reuse the v2.36/v2.37 fold manifest through the v2.37 P0
per-image table.

P0 gate:

```text
image_count == 600
cache_sha_coverage == 100%
mean_delta >= +0.30 dB
hard_delta >= +0.50 dB
easy_delta >= +0.05 dB
p05 >= 0.00 dB
CVaR5 >= -0.01 dB
worst_delta >= -0.05 dB
severe_count == 0
strong_reference_regression_count == 0
fold_pass == 5/5
```

P1 uses out-of-fold alpha selection. For each heldout fold, the largest alpha
that passes the same safety gate on the other four folds is selected and then
evaluated exactly once on the heldout fold. Heldout results may not be used to
retune alpha.

P2 computes a critical-alpha safety margin atlas for the P1-selected alpha. The
projection phase is authorized only when the selected alpha is not close to the
strong-reference or severe failure boundary.

P3 unmasked micro-alpha free-tensor projection is blocked until P0, P1, and P2
all pass. If launched, it may test `S4_plus_S6`, `S6_decoder_early`,
`S4_encoder_late`, and `S4_plus_S5_plus_S6`; it is not a masked/oracle P5.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_38_microalpha_same_context_wdmamba_safe_substrate_projection_20260706/`

## Result

Decision: `P0_FAIL_STOP_NO_MICROALPHA_SAFE_SUBSTRATE`.

P0 ran the planned fine micro-alpha sweep over 600 train-derived full-image
same-context cache rows. No alpha passed the strict no-selector full600/fold
gate. P1 OOF alpha selection, P2 safety margin, P3 unmasked micro-alpha
free-tensor projection, bridge/generator training, canary80, and locked test are
not authorized.
