# Haze4K v2.0 StrongExpert-GainMix Evidence

Date: 2026-06-14

Status: `C0_PLANNED`

Route card: `experience_docx/experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v20-strongexpert-gainmix`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Locked test: blocked and untouched for C0.

## C0 Outputs

Planned outputs:

- `v20_candidate_zoo_manifest.json`
- `v20_candidate_zoo_per_image_metrics.csv`
- `v20_candidate_zoo_single_expert_summary.csv`
- `v20_candidate_zoo_alpha_grid.csv`
- `v20_candidate_zoo_oracle_grid.csv`
- `v20_candidate_zoo_oracle_composition.csv`
- `v20_candidate_zoo_failure_bins.csv`
- `v20_candidate_zoo_decision.md`

## Parallel Evidence Hygiene Outputs

Planned outputs:

- `v37_d8_d9_reconciliation_audit.json`
- `v37_d8_d9_reconciliation_audit.md`
- `v37_d8_d9_reconciliation_inconsistencies.csv`
- `v37_d9_forensic_bucket_summary.csv`
- `v37_d9_forensic_top_regressions.csv`
- `v37_d9_forensic_feature_drift.csv`
- `v37_d9_forensic_summary.json`
- `v37_d9_forensic_summary.md`

These audits are evidence hygiene and failure attribution only. They are not a
DTA-v3.7 locked-test repair path.
