# Haze4K v5 CHD-RM v2c Need Coverage Calibration

Date: 2026-07-09

Status: paused after full val-inner audit.

## Scope

- Route: CHD-RM v5.
- Stage: v2c `R_need` output-scale and coverage calibration audit.
- Branch: `codex/haze4k-v5-v2c-chd-rm-need-coverage-calibration`.
- Source commit: `794467bc84211b4762b6c39f326408beb5e6d829`.
- Runtime: `convir-4090` only.
- Dataset: Haze4K train split through the v1 train_inner/val_inner split.
- Locked Haze4K test: not used.

## Route Identity

This is a v2 repair stage, not v3. It does not connect RARM, train gamma, tune on
the locked test, or change the dehazing output. It tests whether v2b's useful
`R_need` ranking signal is blocked mainly by output-scale compression.

## Method

The v2b heads are frozen. For each variant, train_inner predictions are used to
fit post-hoc monotone or affine calibrators, then val_inner is evaluated under
the same v2/v2b gate:

- identity;
- affine p01-p99;
- affine p10-p90;
- mean/std;
- logit mean/std;
- 1001-point quantile map.

The shuffled-target control is retained.

## Main Evidence

Best ranking variant before calibration:

- `d6c_ordinal_quantile` identity: Pearson 0.3632, Spearman 0.3298, AUROC
  0.7133, coverage 0, false-strong 0, monotonic 4/4.

Best d6c calibrated variants:

| Method | Pearson | Spearman | AUROC | Coverage | False-strong | Monotonic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_std | 0.3490 | 0.3290 | 0.7131 | 0.3159 | 0.1153 | 4/4 |
| quantile_map_1001 | 0.3270 | 0.3298 | 0.7133 | 0.3448 | 0.1287 | 4/4 |
| affine_p10_p90 | 0.3504 | 0.3288 | 0.7130 | 0.3824 | 0.1486 | 4/4 |

Shuffled control remains clearly invalid:

- `d6s_shuffled_quantile` calibrated variants keep negative rank metrics and
  AUROC near 0.337, with very high false-strong rates when coverage is restored.

## Decision

Decision: `PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH`.

Scale calibration alone is not enough. It can restore non-degenerate coverage,
but the restored high-response regions are not safe enough in low-density /
low-need context. Do not run D2, connect RARM, or enter v3 from this evidence.

## Next Step

Repair spatial ranking or head capacity before modulation. The next experiment
should stay inside CHD-RM and target safe high-need localization rather than
global score scaling.
