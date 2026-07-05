# Haze4K v2.29 NoPost ILFRB-ACS Safe OOF Action-Bank Calibration Evidence

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration.md`

Status: `P2A_FAIL_SAFE_OOF_ACTION_BANK_CALIBRATION_PAUSE`

Runtime server: `convir-4090`

Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`

Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked. Training is blocked.

## Key Results

- P2A decision: `P2A_FAIL_SAFE_OOF_ACTION_BANK_CALIBRATION_PAUSE`
- Best safety-envelope variant: `bucket_strength_grid`
- Safety-envelope selected mean/hard/easy: `+0.833159 / +1.375280 / +0.334240`
- Safety-envelope p05/CVaR5/severe: `0.0 / 0.0 / 0.0`
- Failed safety controls: deployable mild unsafe `0.215625`,
  cross-bucket unsafe `0.455556`, hard-to-easy cross severe `0.716667`,
  overstrong 1.5 unsafe `0.441667`
- Best table-policy variant: `energy_norm_plus_bucket_strength`
- Table-policy mean/hard/easy: `+0.077516 / +0.240312 / +0.037175`
- Table-policy p05/CVaR5/severe: `-0.219939 / -0.433256 / 0.0625`
- Table-policy fold-tail pass: `3/5`; table pass: `0`
- Training launched: `False`
- Locked test touched: `False`

Interpretation: bucket-aware strength preserved useful selected-policy OOF
stratification, but plausible cross-bucket and overstrong negative controls
remain unsafe, and the GT-free table policy is too weak. This blocks P2B,
training, and locked-test use.

## Primary Files

- `v229_p0_arch_contract_delta.md`
- `v229_p2a_negative_control_taxonomy.csv`
- `v229_p2a_deployable_candidate_tail_by_bucket_stage.csv`
- `v229_p2a_safety_envelope_variant_summary.csv`
- `v229_p2a_oof_table_policy_report.csv`
- `v229_p2a_oracle_selected_table_policy_gap.md`
- `v229_p2a_cross_bucket_directionality_report.csv`
- `v229_p2a_stage_action_pruning_report.csv`
- `v229_p2a_fold_tail_report.csv`
- `v229_p2a_noop_useful_unsafe_base_rate_report.json`
- `v229_p2a_failonly_diagnostic_probe.csv`
- `v229_p2a_closeout.json`
- `run_v229_p2a.sh`
- `monitor_v229.sh`
- `status.txt`

This directory is compact text evidence only. It excludes checkpoints, weights, images, arrays, archives, and raw feature dumps.
