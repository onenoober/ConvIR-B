# Haze4K v4 Internal Validation Split Protocol

Date: 2026-07-08

## Split

Use only Haze4K `train/haze`.

- `haze4k_train_internal_holdout256.txt`: 256 image names sampled with seed `3407`.
- `haze4k_train_adapter_train.txt`: remaining 2744 image names.
- `haze4k_train_diagnosis_trainfit128.txt`: sorted first 128 image names, retained for legacy comparison only.

## Policy

Future adapter-only training routes must train on `adapter_train` and report both train-fit and internal-holdout metrics. A route may continue only if internal-holdout movement is non-negative under its written gate.

Locked Haze4K test remains blocked until a later route card authorizes exactly one fixed checkpoint and one confirmation command.
