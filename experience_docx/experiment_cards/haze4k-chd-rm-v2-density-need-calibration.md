# Haze4K v5 CHD-RM v2 Density-Need Calibration

Date: 2026-07-08

Status: paused; decision `PAUSE_V2_DUAL_HEAD_NOT_PASSED`.

## Scope

- Route: CHD-RM v5.
- Stage: v2 continuous haze-density and restoration-need calibration.
- Branch: `codex/haze4k-v5-v2-chd-rm-density-need-calibration`.
- Source branch: `codex/haze4k-v5-v1-chd-rm-data-baseline-lock`.
- Source commit: `793d8a3510733debeb7538ff14d70313445fe010`.
- Runtime: `convir-4090` only.
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Baseline checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Split source: v1 `haze4k_internal_split_2400_600.json`.
- Locked test: not used.

## Route Identity

This is a new CHD-RM stage, not a continuation of an older selector, teacher,
post-processing, or color/texture/structure route. The stage does not change the
ConvIR-B dehazing output. It only tests whether ConvIR-B features can support
continuous `H_density` and `R_need` response maps for later residual modulation.

Forbidden in v2:

- no RARM connection;
- no gamma mapper training;
- no Haze4K locked-test command;
- no independent color, luminance, texture, or structure fidelity module;
- no Lab, luminance, gradient, or texture target as the route core;
- no checkpoint or threshold selection from test data.

## Metric Contract

Targets are train-derived paired proxies:

- `H_density_target = normalize_train_p1_p99(blur(gray_abs(I_hazy - I_gt)))`.
- `R_need_target = normalize_train_p1_p99(blur(gray_abs(O_A0 - I_gt)))`.

All thresholds are computed from `train_inner` only. Validation is on the fixed
v1 `val_inner` 600 images. The Haze4K locked test remains blocked.

Gate thresholds:

- density Pearson >= 0.25;
- density Spearman >= 0.30;
- density AUROC high-vs-low >= 0.65;
- need Pearson >= 0.20;
- need Spearman >= 0.25;
- need AUROC high-vs-low >= 0.65;
- density and need calibration bins must be monotonic;
- density and need strong-response coverage must each be between 0.01 and 0.90;
- low-haze false-strong-recovery rate <= 0.10;
- shuffled target control must fail the same gate.

## Authorized v2 Experiments

| ID | Variant | Purpose | Launch status |
| --- | --- | --- | --- |
| V2-D0 | handcrafted dark-channel density proxy | proxy lower bound | authorized |
| V2-D1 | frozen ConvIR res1 feature + density/need head | main feature learnability check | authorized |
| V2-D3 | density-only head | density signal isolation | authorized |
| V2-D4 | need-only head | residual-need signal isolation | authorized |
| V2-D5 | shuffled target dual-head control | false-positive control | authorized |
| V2-D2 | partial-unfreeze encoder-last + head | only if D1 is close to/pass and D5 fails | blocked for first launch |

## Pause Rule

Pause after D0/D1/D3/D4/D5 evidence is written. Continue to v3 only if D1 passes
and D5 fails. Consider D2 only if D3/D4 show clear single-signal learnability but
D1 is narrowly below gate.

## Current Decision

Full train_inner/val_inner evidence has been collected on `convir-4090`.

- V2-D0 handcrafted density proxy failed: density Pearson `0.0249`, Spearman `-0.0103`, AUROC `0.4848`.
- V2-D1 dual head passed density but failed need: density Pearson `0.6715`, Spearman `0.6437`, AUROC `0.8925`; need Pearson `0.1365`, Spearman `0.2198`, AUROC `0.6466`, strong-response coverage `0.0`.
- V2-D3 density-only passed strongly: Pearson `0.6873`, Spearman `0.6628`, AUROC `0.9043`, monotonic pairs `4/4`.
- V2-D4 need-only is close but not passed: Pearson `0.1721`, Spearman `0.2508`, AUROC `0.6648`, strong-response coverage `0.0`.
- V2-D5 shuffled-target control failed clearly: density Pearson `-0.1336`, need Pearson `-0.2081`.

Decision: do not proceed to v3 RARM. Pause v2 and revise/extend the
`R_need` calibration path before any residual modulation connection.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/`
