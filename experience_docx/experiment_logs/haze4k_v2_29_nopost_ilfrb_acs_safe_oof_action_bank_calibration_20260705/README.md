# Haze4K v2.29 NoPost ILFRB-ACS Safe OOF Action-Bank Calibration Evidence

Route card:
`experience_docx/experiment_cards/2026-07-05-haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration.md`

Status: `PLANNED_CLOUD_AUDIT_LOCKED_TEST_BLOCKED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`

Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked. Training is blocked.

## Purpose

v2.29 tests whether the v2.28 OOF action-bank strata can be made safe enough
for future selector work by using safety envelopes, bucket-aware strength,
stage/action pruning, and a GT-free OOF table policy.

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

This directory is compact text evidence only. Do not sync checkpoints, weights,
image outputs, arrays, archives, or raw feature dumps.
