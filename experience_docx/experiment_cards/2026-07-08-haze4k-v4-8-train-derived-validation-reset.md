# Haze4K v4.8 Train-Derived Validation Reset

Date: 2026-07-08

Branch: `codex/haze4k-v4-8-train-derived-validation-reset`

Status: planned cloud K-fold validation.

Route identity: validation-system reset and train-derived OOF audit of the DCFSB-bottleneck adapter recipe. This is not adapter4 promotion, not a locked-test rescue, not an A3 continuation, and not an SDC-Lite expansion.

Parent/source: `github/codex/haze4k-v4-7-dcfsb-candidate-validation` at `1e0ccb97682a2cbd9deda2c10da5e1843f5ccca1`.

Forbidden flows:

- Do not run or enumerate locked Haze4K test.
- Do not run additional locked-test commands for adapter4.
- Do not tune model structure, thresholds, checkpoints, epochs, or variants from v4.7 locked-test results.
- Do not continue A3, expand v4.5 SDC-Lite, or connect an uncalibrated R field to skip/FAM/restoration outputs.
- Do not treat a single internal_holdout split as promotion evidence.

## Phase 1: K-fold/Tail-safe Validation

Goal: test whether the v4.6 DCFSB-bottleneck adapter recipe is stable under train-derived out-of-fold validation.

Recipe: train DCFSB-bottleneck adapter-only from official A0 on each train-derived fold, with the same v4.6 adapter4 budget: `stop_epoch=4`, `lr=1e-4`, `weight_decay=1e-4`, seed `3407`, `valid_freq=999`, and no default validation.

Split: Haze4K train only, grouped by base image id and balanced by train-derived proxies. Locked test is not touched.

OOF gate:

- OOF mean dPSNR >= `+0.025`
- OOF positive ratio >= `0.55`
- OOF median dPSNR > `0`
- OOF p5 dPSNR >= `-0.25`
- OOF p1 dPSNR >= `-0.50`
- bootstrap 95% CI low > `0`
- sign-test one-sided p < `0.01`
- mean dHighL1 <= `+0.000005`
- every fold mean dPSNR > `0`
- every fold positive ratio >= `0.50`
- every proxy bin mean dPSNR >= `-0.005`
- every proxy bin positive ratio >= `0.50`
- low-saturation subgroup must not reproduce v4.7 negative mean / low win rate
- locked_test_touched=false and test_split_enumerated=false

## Phase 2: R-only Calibration

R-only calibration is allowed only as a separate probe after Phase 1 is written or explicitly marked independent. It must not alter restoration output and must not touch locked test.


## Results (2026-07-08)

Status: completed gate fail.

Phase 1 OOF used five train-derived folds, with all folds rerun under a predeclared `batch_size=7` engineering repair after the original `batch_size=8` launch exposed final-batch-size-1 BatchNorm failures in fold2/fold4. Locked test was not touched or enumerated.

OOF summary: mean dPSNR `0.051192`, positive ratio `0.624667`, median `0.050976`, p5 `-0.266470`, p1 `-0.469765`, bootstrap CI low `0.041783`, sign-test p `3.3161829e-43`, mean dHighL1 `0.0000017954`. The route failed the full v4.8 tail-safe gate on `oof_p5_delta_psnr, every_proxy_bin_positive_ratio, low_saturation_subgroup`. The low-saturation q1 subgroup remained diffuse-risk: mean dPSNR `0.010802`, positive ratio `0.488000`.

Phase 2 R-only calibration audited the existing v4.5 SDC-Lite R field on the same 3000 train-derived OOF union. It failed: R std mean `0.081058`, corr(R,input-GT L1) `-0.438983`, corr(R,A0 error proxy) `-0.421714`, heavy-haze q4 vs q1 relative response `-0.059567`, low-saturation q1 minus high-saturation q4 R mean `-0.022944`.

Decision: do not promote the DCFSB adapter recipe, do not run locked test, do not expand v4.5/SDC-Lite, and do not connect this R field to skip/FAM/restoration outputs. Future work needs a new calibrated conditional routing signal, not more tuning of adapter4 or uncalibrated R.
