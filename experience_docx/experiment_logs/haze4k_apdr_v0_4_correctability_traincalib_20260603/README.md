# APDR-v0.4 Correctability Train Calibration

Date: 2026-06-03

Status: completed diagnostic; gate passed.

Route card: `experience_docx/experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md`

## Command

```bash
bash experience_docx/experiment_logs/haze4k_apdr_v0_4_correctability_traincalib_20260603/run_apdr_v0_4_correctability_traincalib.sh
```

Executed on AutoDL `autodl-dehaze3` under:

```text
/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-4-cclf-diagnostics
```

## Artifacts

- `correctability_traincalib_apdr_v0_4_correctability_traincalib_seed3407.json`
- `correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_seed3407.csv`
- `correctability_traincalib_test_apdr_v0_4_correctability_traincalib_seed3407.csv`
- `correctability_traincalib_history_apdr_v0_4_correctability_traincalib_seed3407.csv`
- `correctability_traincalib_apdr_v0_4_correctability_traincalib_seed3407.log`
- `status.txt`

## Results

| Check | Observed | Required | Result |
| --- | ---: | ---: | --- |
| train-calibrated tau | `0.9897` | pass constraints | pass |
| train OOF AUC | `0.99997` | high ranking quality | pass |
| train OOF Spearman | `0.9798` | high ranking quality | pass |
| train OOF easy top-25 open rate | `0.0493` | `<= 0.05` | pass |
| train OOF negative false-open | `0.0` | `<= 0.02` | pass |
| train OOF positive-hard recall | `0.9691` | `>= 0.95` | pass |
| test AUC | `1.0` | high ranking quality | pass |
| test Spearman | `0.9729` | high ranking quality | pass |
| test easy top-25 open rate | `0.0200` | `<= 0.05` | pass |
| test negative false-open | `0.0` | `<= 0.02` | pass |
| test positive-hard recall | `0.9750` | `>= 0.95` | pass |

## Decision

Decision label: `PASS_TRAIN_CALIBRATED_CORRECTABILITY`.

The train-calibrated threshold is safe enough to carry into a separate v0.4A low-field-only branch. This does not authorize the full low+color v0.4C route because the color free-parameter gate failed.
