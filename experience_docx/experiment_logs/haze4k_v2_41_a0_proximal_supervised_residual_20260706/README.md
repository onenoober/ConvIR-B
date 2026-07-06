# Haze4K v2.41 A0-Proximal Supervised Residual Evidence

Status: `P0_STAGE0_PREFLIGHT_PASS_CANARY32_WRITTEN`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-41-a0-proximal-supervised-residual.md`

Central index path:
`experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-41-a0-proximal-supervised-residual`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked.

## Evidence Files

Compact sync candidates:

- `status.txt`
- `run_v241_p0_stage0_preflight.sh`
- `v241_p0_stage0_preflight.json`
- `v241_p0_closeout.json`
- `run_v241_canary32_oof.sh`
- `v241_canary32_oof_summary.json`
- `v241_canary32_oof_folds.csv`
- `v241_canary32_oof_epoch_history.csv`
- `v241_canary32_oof_per_image.csv`
- `v241_canary32_closeout.json`

## Metric Contract

P0 Stage-0 proves strict official-checkpoint partial load, zero-init/no-op
equivalence, finite forward, no forbidden postprocess symbols, and locked-test
blocked status. It does not train, run canary80, or touch locked test.

Canary32 OOF is authorized only after P0 pass. It trains only `A0PROX_*` on
five train-derived folds (`32` train images and disjoint `32` held-out images
per fold) and freezes all official ConvIR-B parameters. It does not use teacher
targets, selectors, canary80, or locked test.

Canary32 passes only if global mean/hard/easy/p05/CVaR5/tail gates pass,
`4/5` folds pass the same gate, easy residual energy is at most `0.50` of hard
residual energy, and hinge violation rate does not worsen across training.

## Result

P0 Stage-0 preflight passed at route commit `7c27b93`.

Key P0 facts:

- checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`;
- strict partial load loaded `602` official keys;
- missing keys were only `A0PROX_beta` and `A0PROX_head.*`;
- synthetic output shapes were `64/128/256`;
- identity max absolute difference vs official ConvIR-B was `0.0`;
- forbidden postprocess symbol hits were `0`;
- locked test was untouched.

Canary32 OOF status: written, not yet launched in this evidence snapshot.
