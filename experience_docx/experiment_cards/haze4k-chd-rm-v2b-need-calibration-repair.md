# Haze4K v5 CHD-RM v2b Need Calibration Repair

Date: 2026-07-08

Status: prepared; cloud execution pending.

## Scope

- Route: CHD-RM v5.
- Stage: v2b `R_need` target calibration and feature audit.
- Branch: `codex/haze4k-v5-v2b-chd-rm-need-calibration-repair`.
- Source branch: `codex/haze4k-v5-v2-chd-rm-density-need-calibration`.
- Source commit: `e8112c508cae3033728969315f9f1a813af46bdd`.
- Runtime: `convir-4090` only.
- Dataset: Haze4K train split through the v1 train_inner/val_inner split.
- Locked Haze4K test: not used.

## Route Identity

This is a v2 repair stage, not v3. It does not connect RARM, train gamma, tune
on the locked test, or change the dehazing output. The only purpose is to decide
whether `R_need` failed because of target calibration and head formulation.

## Authorized Experiments

| ID | Variant | Purpose |
| --- | --- | --- |
| R0 | target audit | quantify raw need dynamic range, coverage, density relation, and A0 metric relation |
| D6a | quantile target | align high-need target with empirical train_inner CDF |
| D6b | log target | test whether raw need needs log compression before normalization |
| D6c | ordinal quantile head | use q20/q33/q66/q80 ranking labels to repair amplitude collapse |
| D6s | shuffled quantile control | ensure repair does not pass under mismatched targets |

## Gate

- need Pearson >= 0.20;
- need Spearman >= 0.25;
- need AUROC high-vs-low >= 0.65;
- need monotonic pairs = 4/4;
- need strong-response coverage in [0.01, 0.90];
- low-density/low-need false-strong rate <= 0.10;
- shuffled target control must fail.

## Pause Rule

Pause after R0/D6a/D6b/D6c/D6s. Do not run v3 unless a repair variant passes and
the shuffled control fails. Do not run D2 before this evidence is interpreted.

