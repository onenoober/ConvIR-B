# APDR-v0.2RC Diagnostics Sync Manifest

Date: 2026-06-03

## Code Changes

| Path | Synchronized role |
| --- | --- |
| `Dehazing/ITS/main.py` | Adds APDR residual capacity, global budget calibration arguments, delta loss weight, and configurable grad clipping. |
| `Dehazing/ITS/train.py` | Adds `apdr_residual_only` training scope, APDR delta supervision loss, and configurable grad clipping. |
| `Dehazing/ITS/models/APDRConvIR.py` | Passes residual capacity into APDR adapters and records delta-supervision regularization terms. |
| `Dehazing/ITS/models/apdr_modules.py` | Adds optional shallow residual body and power-adjusted global budget calibration for APDR-v0.2R/RC. |
| `experience_docx/tools/preflight_haze4k_apdr_v0_2rc_budget.py` | Extends v0.2RC replay with oracle variants, BCE deciles, leakage and coverage tables, spatial bottleneck summaries, and optional selector checkpoint save. |
| `experience_docx/tools/eval_haze4k_checkpoint_compare.py` | Allows APDR checkpoint evaluation to rebuild candidates with `apdr_residual_capacity`. |

## New Experiment Launchers

| Path | Purpose |
| --- | --- |
| `experience_docx/experiment_logs/haze4k_apdr_v0_2rc_oracle_diagnostic_20260603/run_apdr_v0_2rc_oracle_diagnostic.sh` | Runs non-training oracle-on-fail diagnostics for the selected v0.2RC budget. |
| `experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/run_apdr_v0_2rc_frozen_residual_stop20.sh` | Saves a selector checkpoint only after action/oracle prechecks, then runs a frozen-selector residual-only stop20 scout. |
| `experience_docx/experiment_logs/haze4k_apdr_v0_2rc_delta_residual_20260603/run_apdr_v0_2rc_delta_residual_stop20.sh` | Runs a delta-supervised residual-only stop20 scout using the saved selector checkpoint. |
| `experience_docx/experiment_logs/haze4k_apdr_v0_2rc_residual_capacity_20260603/run_apdr_v0_2rc_residual_capacity_stop20.sh` | Runs a shallow residual-capacity stop20 scout using the saved selector checkpoint. |

## Expected Text Evidence

When the AutoDL run finishes, only text-readable outputs should be copied back
or staged for GitHub:

- `gate_*.json`
- `budget_summary_*.json`
- `oracle_variant_summary_*.json`
- `oracle_per_image_*.csv`
- `bce_deciles_*.csv`
- `budget_leakage_easy_top25_*.csv`
- `hard_coverage_bottom25_*.csv`
- `spatial_bottleneck_*.csv`
- `scout_eval_compare_*.json`
- `scout_eval_bucket_analysis_*.json`
- `scout_eval_per_image_*.csv`
- launcher, status, train, and tmux text logs

Do not stage:

- `selector_checkpoint_*.pkl`
- `Best.pkl`, `Last.pkl`, or any model checkpoint
- images, datasets, arrays, or raw inference folders
