# v2.16 WLDB-A Training Protocol

Status: `PLANNED_AFTER_T2_PASS`

This stage is authorized only because v2.16 T0/T1/T2 passed. It remains
train-derived and does not touch locked Haze4K.

## Scope

WLDB-A:

- official ConvIR-B anchor initialized from `haze4k-base.pkl`;
- insert zero-init `nopost_wldb` after `Decoder[2]`, before `feat_extract[5]`;
- freeze all official ConvIR-B parameters;
- train only `nopost_wldb.*`;
- use Haze4K train-derived fold split from `v216_t1_per_image_band_deltas.csv`;
- fold `0` is validation, folds `1-4` are training;
- no WD0375 loss, teacher output, expert output, RGB post-correction, or locked
  test command.

## First Screen

Run one seed first:

- seed: `3407`;
- epochs: `20`;
- batch size: `8`;
- crop size: `256`;
- optimizer: AdamW on `nopost_wldb.*` only;
- losses: GT final L1, LL lowband L1, small A0-preserve L1, and action-budget
  hinge against A0 on train-derived crops.

Evaluate `model_5`, `model_10`, `model_15`, `model_20`, `Best`, and `Final` on
fold0 against A0.

Pass requires all:

- mean dPSNR >= `+0.05`;
- hard bottom25 dPSNR >= `+0.10`;
- easy top25 dPSNR >= `-0.05`;
- positive ratio >= `0.55`;
- severe loss count (`dPSNR <= -0.20`) <= `12/480`;
- strong-reference regressions (`A0 top25`, dPSNR <= `-0.05`) <= `48/120`.

Pass decision: `WLDB_A_SCREEN_PASS_ALLOW_MULTI_SEED`.

Fail decision: `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`.
