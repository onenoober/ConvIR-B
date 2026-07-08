# Haze4K v5 CHD-RM v2b Need Calibration Repair

Date: 2026-07-09

Status: paused after full val-inner audit.

## Scope

- Route: CHD-RM v5.
- Stage: v2b `R_need` target calibration and feature audit.
- Branch: `codex/haze4k-v5-v2b-chd-rm-need-calibration-repair`.
- Source commit: `e8112c508cae3033728969315f9f1a813af46bdd`.
- Runtime: `convir-4090` only.
- Dataset: Haze4K train split through the v1 train_inner/val_inner split.
- Locked Haze4K test: not used.

## Route Identity

This is a v2 repair stage, not v3. It does not connect RARM, train gamma, tune
on the locked test, or change the dehazing output. The only purpose is to decide
whether `R_need` failed because of target calibration and head formulation.

## Experiments

| ID | Variant | Purpose |
| --- | --- | --- |
| R0 | target audit | quantify raw need dynamic range, coverage, density relation, and A0 metric relation |
| D6a | quantile target | align high-need target with empirical train_inner CDF |
| D6b | log target | test whether raw need needs log compression before normalization |
| D6c | ordinal quantile head | use q20/q33/q66/q80 ranking labels to repair amplitude collapse |
| D6s | shuffled quantile control | ensure repair does not pass under mismatched targets |

## Main Evidence

| Variant | Target | Pearson | Spearman | AUROC | Pred high coverage | Monotonic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| d6a_quantile | quantile | 0.2710 | 0.2465 | 0.6611 | 0.0000 | 3/4 |
| d6b_log | log | 0.2712 | 0.2914 | 0.6883 | 0.0000 | 4/4 |
| d6c_ordinal_quantile | quantile | 0.3632 | 0.3298 | 0.7133 | 0.0000 | 4/4 |
| d6s_shuffled_quantile | quantile | -0.2487 | -0.2530 | 0.3369 | 0.0000 | 0/4 |

## Decision

Decision: `PAUSE_V2B_NEED_REPAIR_NOT_PASSED`.

`R_need` ranking is not dead: D6c has real signal and the shuffled control
fails. However, all positive variants still have degenerate strong-response
coverage (`0.0`), so `R_need` cannot be connected to RARM yet.

## Next Step

Run v2c scale/coverage calibration before any D2, RARM, or v3 expansion.
