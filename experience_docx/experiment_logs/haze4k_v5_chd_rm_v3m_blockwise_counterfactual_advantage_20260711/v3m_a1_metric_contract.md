# v3m A1 Block16 Local-Actuation Observability Contract

Date: 2026-07-11

Status: `COMPLETED_GATE_PASS`.

## Scope

A1 asks only whether the frozen block16 five-level oracle action has an
already-deployable local observability signal. It is not a controller, router,
threshold calibration, or policy replay experiment. It does not train or save
weights, and it uses Haze4K train-derived OOF only.

The block16 target is recomputed from the frozen `D_ref`/`D_rep` output step on
the exact common ladder `{0, 0.125, 0.25, 0.5, 1.0}`. A block is positive only
when its oracle-selected action is greater than `0.125`.

## Pinned Inputs

| Artifact | SHA256 |
| --- | --- |
| fresh train-derived split manifest | `c8c00fefc965ded3389b6311fc67ea521e1f3174f27793688544abe09dc420e7` |
| v3l A0 closeout | `2ca39ec1e17f4be794121603e3392a4e042e4d93b0e823454f7cf539f172d05d` |
| v3l frozen operator manifest | `1d2ffa499128ad08a272d67c5439583900afe8ef87fb3256193ad5fe21c3af84` |
| v3m fixed-alpha OOF reference rows | `b4a10184fab77b0045440dc88530d000a892acf2105a6295d5ad8a488c67ecb1` |
| v3m A0 source manifest | `8966996c9c93f6f2f3fbdda536b69ea6aa03e1bf5432f127de47ca8ea95dd8a5` |
| official Haze4K base checkpoint | `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088` |
| frozen control checkpoint | `08207119a5cf9e5c439dd2cb81b99029ade1861f2739d31e75f2f9f78d57c0f2` |
| frozen density artifact | `1ffce13dccb41d96a47c2b5275f87bf2fdb73c226a190cfa240e5c71c1ec326f` |
| frozen D7c artifact | `09f449232024395cf64db15a2a0efa0f12d3e0e049e1da3d67229a3dc5729361` |

The runtime must reproduce each image's `FIXED_ALPHA_0.125` PSNR delta from
the v3m A0 raw OOF table with maximum absolute difference `<= 1e-6 dB` before
any A1 signal metric is valid.

## Signals And Metric

All four signals have fixed high-score direction; no sign flip, feature
selection, weighting, threshold, or fitted calibration is allowed:

1. D7c score mean in the output block;
2. mean squared frozen direct-output step;
3. fixed product of D7c score and direct-step energy;
4. alpha1 pre-clamp exposure fraction.

For each image/operator/signal, calculate AUROC for the block-positive label.
Images without both positive and negative blocks have undefined AUROC and are
not included in the grouped mean, but their count remains reported. Bootstrap
the mean image AUROC with 4,000 deterministic draws, seed `3407`.

## Stages And Gate

The required engineering screen is the first 32 manifest-ordered OOF images for
both operators. It passes only if all 64 fixed-alpha replays meet `1e-6 dB` and
the cloud-only block table is complete. Its result cannot evaluate the science
gate.

The formal audit uses all 1,200 OOF images. A signal passes only if, for both
operators, it has valid positive/negative blocks in at least 80% of images and
grouped mean AUROC CI95 low `>= 0.56`. The same predeclared signal must pass for
both operators.

Formal pass records
`V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`.
Formal fail records
`V3M_A1_LOCAL_ACTION_OBSERVABILITY_WEAK_STOP_NO_CONTROLLER`.

No outcome authorizes controller training, a learned ranker, threshold or
route-confirm selection, canary, physics/proxy policy work, or locked-test
access. A formal pass authorizes only a separately contracted A2 OOF
calibration audit.

## Smoke Correction

The first 32-image smoke stopped on its first fixed-alpha replay because the
custom A1 reader omitted the random/NumPy/Torch/CUDA seed and cuDNN
deterministic settings used by the v3l-A1 reference run. No image completed,
the partial cloud-only table contains only its header, and no A1 metric was
read. Smoke r1 restored the initialization but still recomputed OOF folds from
the 32-image subset, which selects different frozen fold heads. Smoke r2 uses
the exact 1,200-image fold map before taking the deterministic 32-image prefix.
Each failed smoke output remains cloud-only as `FAILED_ENGINEERING`.

## Smoke r2 Result

Smoke r2 used the full 1,200-name fold assignment before selecting its 32-name
prefix. `D_ref` and `D_rep` both reproduced fixed `alpha=0.125` exactly with
maximum difference `0 dB`; the 40,000 block records and four compact output
artifacts were complete. The smoke has no scientific AUC decision and did not
select a signal. It authorizes only the formal 1,200-image OOF run.

## Formal Result

All 2,400 fixed-alpha replays were exact. Direct-step energy passed for both
operators (AUROC CI95 low `0.8522` / `0.8516`); D7c score and fixed score-times-
energy also passed, while clip exposure did not. The decision is
`V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`.
