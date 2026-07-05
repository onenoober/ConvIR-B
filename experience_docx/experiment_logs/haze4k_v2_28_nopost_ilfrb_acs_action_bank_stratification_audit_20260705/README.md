# Haze4K v2.28 NoPost ILFRB-ACS Action-Bank Stratification Audit Evidence

Route card:
`experience_docx/experiment_cards/2026-07-05-haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit.md`

Status: `PLANNED_CLOUD_AUDIT_LOCKED_TEST_BLOCKED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`

Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked. This audit uses train-derived samples only.

## Purpose

v2.27 proved strong same-sample ILFRB-ACS oracle capacity, but failed P2 because
the action bank did not create real no-op or unsafe strata. v2.28 replaces the
same-sample action replay with out-of-fold prototype replay, cross-bucket swap
diagnostics, and explicit negative controls.

## Primary Files

- `v228_p0_arch_contract_delta.md`
- `v228_p2a_oof_prototype_action_bank_replay.csv`
- `v228_p2a_action_preference_by_bucket.csv`
- `v228_p2a_strength_safety_curve.csv`
- `v228_p2a_noop_unsafe_base_rate_report.json`
- `v228_p2a_cross_sample_swap_matrix.csv`
- `v228_p2a_oracle_vs_oof_gap.md`
- `v228_p2a_fold_tail_report.csv`
- `v228_p2b_probe_feature_ablation.csv`
- `v228_closeout.json`
- `run_v228_p2a.sh`
- `monitor_v228.sh`
- `status.txt`

This directory is intended for compact text evidence only. Do not sync
checkpoints, weights, image outputs, arrays, archives, or raw feature dumps.
