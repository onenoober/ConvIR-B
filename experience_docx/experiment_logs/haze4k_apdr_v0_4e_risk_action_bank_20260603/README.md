# APDR-v0.4E Risk Action Bank Intermediate Audit

Date: 2026-06-03

Status: completed on `autodl-dehaze4`; E0 locked-threshold confirmation passed.

## Purpose

This audit is the first APDR-v0.4E step after the v0.4D spatial probe. It keeps
the v0.4D confidence thresholds locked, evaluates them on an independent
confirm slice, and writes the intermediate tables needed before any OOF
calibration or deployable selector work.

It does not train ConvIR-B, does not train a residual head, does not launch
stop20, and does not re-sweep thresholds.

## Default Split

```text
fit slice:     train indices 0..127
confirm slice: train indices 256..383
```

The fit slice reproduces the shallow action generators. The confirm slice is
kept separate from the same-split v0.4D mini-val sweep.

## Command

Run on `autodl-dehaze4`:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/launch_apdr_v0_4e_risk_action_bank_tmux.sh
```

Direct run:

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/run_apdr_v0_4e_risk_action_bank_sigma3.sh
```

## Outputs

- `v04e_locked_threshold_confirm_summary.json`
- `v04e_candidate_action_table.csv`
- `v04e_candidate_action_per_image_sigma3.csv`
- `v04e_risk_feature_auc.csv`
- `v04e_oof_calibration_curve.csv`
- `v04e_accepted_vs_rejected_groups.csv`
- `v04e_strong_failure_signature.csv`
- `v04e_risk_action_bank_apdr_v0_4e_risk_action_bank_sigma3_seed3407.log`
- `status.txt`

## Run Status

The first run exposed a table-writing bug after evaluation and was fixed before
the second run. The second run completed successfully:

```text
start_v0_4e_risk_action_bank ... 2026-06-03T21:20:40+08:00
complete_v0_4e_risk_action_bank apdr_v0_4e_risk_action_bank_sigma3_seed3407 2026-06-03T21:28:42+08:00
exit_code=0
```

Default run sizes:

```text
fit_open_count=90
confirm_count=128
confirm_open_count=88
```

## Gate E0

Locked rules:

```text
Rule A: global_plus_spatial_kenel_knn_9, K16, pred_abs_mean >= 0.010107760690152645
Rule B: spatial_priors_ridge_10, K16, pred_abs_mean >= 0.01394791239872575
```

Pass line:

```text
severe_regressions = 0
strong_regressions <= 1 per 128
easy_top25_gain >= -0.02 dB
mean_gain >= +0.05 dB
hard_bottom25_gain >= +0.25 dB
keep_count >= 15/128
```

## E0 Result

Both locked rules passed on confirm indices `256..383` without re-sweeping the
threshold.

| Rule | Keep | Mean gain | Hard gain | Easy gain | Strong/severe | L1 drop | Oracle recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule A: `global_plus_spatial_kenel_knn_9`, K16 | `29/128` | `+0.1546 dB` | `+0.3251 dB` | `+0.0562 dB` | `0/0` | `0.1282` | `0.1706` |
| Rule B: `spatial_priors_ridge_10`, K16 | `45/128` | `+0.2141 dB` | `+0.4528 dB` | `+0.0625 dB` | `1/0` | `0.1029` | `0.2363` |

Apply-all candidate rows remain unsafe. For example, full-scale
`spatial_priors_ridge_10` has mean `+0.2760 dB` and hard `+0.5179 dB`, but
strong/severe `8/6` and easy `-0.0346 dB`. The passed signal is therefore the
locked no-op fallback, not broad action application.

Decision:

```text
E0_PASS_AUTHORIZE_OOF_CALIBRATION_ONLY
```

Next allowed step: write/run the APDR-v0.4E 5-fold OOF calibration audit. Still
blocked: full spatial router training, local correction, dense residual heads,
and stop20.
