# Haze4K v2.13 NoPost Feature-Gated Adapter Evidence

Status: `PLANNED_N0_N1_N2_FIRST_LOCKED_TEST_UNTOUCHED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Data:
`/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`

A0 checkpoint:
`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`

Split source:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v24-c12-wd0375-distill/experience_docx/experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/v24_c12_split_manifest.json`

WD0375 teacher cache:
`/sda/home/wangyuxin/ConvIR-B/runtime_cache/v24_c12_wd0375_teacher`

Locked Haze4K test: untouched and blocked.

## Stage Outputs

N0 contract audit:

- `v213_n0_nopost_contract.md`
- `v213_n0_source_audit.json`
- `v213_n0_forbidden_symbol_scan.txt`
- `v213_n0_forward_signature.json`

N1 feature separability:

- `v213_n1_feature_table_manifest.json`
- `v213_n1_oof_gain_risk_probe.csv`
- `v213_n1_feature_ablation_report.csv`
- `v213_n1_calibration_report.json`
- `v213_n1_decision.md`

N2 identity:

- `v213_n2_identity_summary.json`
- `v213_n2_identity_per_image.csv`
- `v213_n2_param_groups.json`
- `v213_n2_state_dict_load_report.txt`

N3 microfit:

- `v213_n3_microfit16_history.csv`
- `v213_n3_microfit64_history.csv`
- `v213_n3_microfit256_history.csv`
- `v213_n3_microfit_leaderboard.csv`
- `v213_n3_gate_action_stats.csv`
- `v213_n3_decision.md`

N4 staged screen:

- `v213_n4_fold_seed_summary.csv`
- `v213_n4_gate_action_distribution.csv`
- `v213_n4_failure_taxonomy.csv`
- `v213_n4_decision.md`

Raw feature tables, checkpoints, images, and large runtime outputs remain
cloud-only by default.
