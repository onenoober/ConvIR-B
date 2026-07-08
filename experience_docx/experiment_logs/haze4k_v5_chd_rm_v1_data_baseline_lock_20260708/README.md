# Haze4K v5 CHD-RM v1 Data Baseline Lock Evidence

Date: 2026-07-08

Status: data manifest stage running.

## Purpose

Lock Haze4K data, train-derived validation splits, file hashes, and leakage
status before any CHD-RM density/need calibration or RARM work.

## Locked-Test Status

No locked-test tuning is allowed. This stage may inspect official test filenames
and hashes only for leakage accounting and final split integrity.

## Files

| File | Role |
| --- | --- |
| `run_v1_data_manifest.sh` | durable command script for data manifest generation |
| `build_v1_data_manifest.py` | manifest/split/leakage generator |
| `haze4k_manifest_train.csv` | train manifest |
| `haze4k_manifest_test.csv` | test manifest |
| `haze4k_internal_split_2400_600.csv` | fixed train-inner/val-inner split |
| `haze4k_oof_folds.csv` | fixed 5-fold OOF table |
| `haze4k_file_hash_summary.json` | hash/count summary |
| `leakage_audit.json` | train/test leakage audit |
| `split_policy.md` | split policy and locked-test policy |
| `data_manifest_summary.md` | compact result summary |
| `decision_record.md` | stage decision so far |

## Closeout Summary

- Decision: `COMPLETED_V1_GATE_PASS`
- A0 val_inner mean PSNR: `39.253655`
- A0 val_inner mean SSIM: `0.995124812`
- Metric reproducibility pass: `True`
- Params: `8630665`
- FPS pass1: `30.914`
- Locked test: not used for tuning or scoring.
