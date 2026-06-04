# APDR-v0.4 Sigma-3 Alignment Diagnostics

Date: 2026-06-03

Status: completed parallel diagnostics on `autodl-dehaze3`.

## Purpose

The v0.4 cache-scale sweep found sigma `3.0` as the strongest lowpass oracle
target on train128, but the stronger free-parameter low and train-calibrated
correctability evidence currently uses sigma `7.0`.

These diagnostics align sigma `3.0` before any v0.4A target switch. They do not
authorize stop20 by themselves and should not change the main v0.4A target away
from sigma `7.0` unless the resulting JSON/CSV evidence is reviewed.

## Commands

Run both jobs on AutoDL in detached tmux panes or sessions:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/run_apdr_v0_4_freeparam_low_sigma3_32.sh
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/run_apdr_v0_4_correctability_traincalib_sigma3.sh
```

Or launch both in one detached tmux session:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/run_apdr_v0_4_sigma3_alignment_parallel.sh
```

## Artifacts

- `freeparam_lowcolor_apdr_v0_4_freeparam_low_sigma3_32_seed3407.json`
- `freeparam_lowcolor_per_image_apdr_v0_4_freeparam_low_sigma3_32_seed3407.csv`
- `freeparam_lowcolor_history_apdr_v0_4_freeparam_low_sigma3_32_seed3407.csv`
- `freeparam_apdr_v0_4_freeparam_low_sigma3_32_seed3407.log`
- `correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.json`
- `correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv`
- `correctability_traincalib_test_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv`
- `correctability_traincalib_history_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv`
- `correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.log`
- `status.txt`

## Decision Use

| Diagnostic | Continue condition | If it fails |
| --- | --- | --- |
| sigma-3 free-parameter low | recovery/correlation/safety stay comparable to sigma `7.0` | keep sigma `7.0` as v0.4A first target |
| sigma-3 correctability traincalib | train-calibrated easy open and hard recall stay safe | do not use sigma `3.0` gate for deployable training |

## Results

| Diagnostic | Verdict | Key observations |
| --- | --- | --- |
| sigma-3 free-parameter low | partial pass signal; strict loss-drop gate failed | loss-drop `0.6891 < 0.80`; oracle recovery `1.0551`; corr `0.9309`; hard bottom-25 gain `+1.2639 dB`; easy top-25 gain `+0.4701 dB`; strong/severe regressions `0` |
| sigma-3 correctability traincalib | pass | train OOF AUC `0.99997`; tau `0.99399`; test AUC `1.0`; test Spearman `0.9701`; test easy open `0.012`; negative false-open `0.0`; positive-hard recall `0.9600` |

## Decision

Sigma `3.0` now has target/application/gate alignment evidence. It is eligible
for the v0.4A LowFieldNet Gate A/B overfit32 preflight. This still does not
authorize stop20: the deployable predictor must pass no-op/cache exactness,
overfit32 learnability, and train128/mini-val gates first.

This root is text-only. Do not commit tensor caches, checkpoints, image outputs,
NumPy arrays, or datasets.
