# APDR-v0.4E OOF Calibration Audit

Date: 2026-06-03

Status: completed on `autodl-dehaze4`; E1 OOF gate failed.

## Purpose

This is the E1 audit after v0.4E E0 passed. It creates 5 stratified folds ove
the Haze4K train split. For each fold, the script derives the K16 low-field
basis and fits the candidate mappers on the other four folds, then evaluates
only the held-out fold.

It writes true OOF candidate-action and risk-calibration tables. It does not
train ConvIR-B, does not train an APDR residual head, and does not launch
stop20.

## Command

Run on `autodl-dehaze4`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/launch_apdr_v0_4e_oof_calibration_tmux.sh
```

Direct run:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/run_apdr_v0_4e_oof_calibration_sigma3.sh
```

## Outputs

- `v04e_oof_locked_threshold_summary.json`
- `v04e_oof_locked_threshold_by_fold.csv`
- `v04e_oof_fold_assignments.csv`
- `v04e_oof_candidate_action_table.csv`
- `v04e_oof_candidate_action_per_image_sigma3.csv`
- `v04e_oof_risk_feature_auc.csv`
- `v04e_oof_calibration_curve.csv`
- `v04e_oof_accepted_vs_rejected_groups.csv`
- `v04e_oof_strong_failure_signature.csv`
- `v04e_oof_policy_search_sigma3.csv`
- `v04e_oof_policy_search_best_by_group_sigma3.csv`
- `v04e_oof_policy_search_summary_sigma3.json`
- `v04e_oof_policy_search_sigma3.log`
- `v04e_oof_calibration_apdr_v0_4e_oof_calibration_sigma3_seed3407.log`
- `status.txt`

## Run Status

The OOF audit completed successfully:

```text
start_v0_4e_oof_calibration ... 2026-06-03T21:50:26+08:00
complete_v0_4e_oof_calibration apdr_v0_4e_oof_calibration_sigma3_seed3407 2026-06-03T22:39:02+08:00
exit_code=0
```

Fold sizes:

```text
active_open_count=1324
fold0 eval=665 open=292
fold1 eval=632 open=279
fold2 eval=592 open=263
fold3 eval=563 open=249
fold4 eval=548 open=241
```

## Gate E1

Locked Rule A/B are evaluated without re-sweeping thresholds.

Pass line:

```text
OOF severe = 0
OOF strong rate <= 1%
OOF easy_top25_gain >= -0.02 dB
OOF hard_bottom25_gain >= +0.25 dB
OOF mean_gain > 0
accepted coverage >= 10%
accepted oracle recovery >= 0.15
```

## E1 Result

Locked Rule A/B failed the OOF gate.

| Rule | Keep | Coverage | Mean gain | Hard gain | Easy gain | Strong/severe | Oracle recovery | Main blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule A: `global_plus_spatial_kenel_knn_9`, K16 | `239/3000` | `0.0797` | `+0.0749 dB` | `+0.2596 dB` | `+0.0000 dB` | `0/5` | `0.1654` | severe `5`, coverage `<10%` |
| Rule B: `spatial_priors_ridge_10`, K16 | `150/3000` | `0.0500` | `+0.0378 dB` | `+0.1352 dB` | `+0.0000 dB` | `0/1` | `0.0835` | severe `1`, hard/recovery/coverage fail |

Apply-all candidate rows remain unsafe despite positive mean/hard movement. The
best apply-all OOF mean row, `convir_spatial_kenel_knn_9` scale `1.0`, has
mean `+0.1602 dB` and hard `+0.4496 dB`, but strong/severe `1/11`.

## Policy Search

A post-hoc low-capacity threshold search over the OOF per-image table retained
`4000` candidate policies and found no gate-passing policy.

Best retained policy:

```text
mapper=spatial_priors_ridge_10
scale=1.0
primary=weighted_residual_norm >= 0.009209617488433332
secondary=nn_distance <= 11.120915794372559
keep=263/3000
coverage=0.0877
mean=+0.0792 dB
hard=+0.2527 dB
easy=+0.0000 dB
strong/severe=0/0
oracle_recovery=0.1748
```

This best policy removes the severe tail and keeps hard gain, but misses the
pre-registered `coverage >= 10%` line. Treat it as diagnostic evidence only,
not as an E2 or deployment authorization.

Decision:

```text
E1_FAIL_STOP_CURRENT_LOCKED_THRESHOLDS
```

Current v0.4E locked thresholds should not proceed to E2, full spatial router,
local correction, dense residual training, or stop20. The remaining useful
signal is a possible future route that pre-registers a stricter safe-subset
policy and evaluates it on a fresh held-out split.
