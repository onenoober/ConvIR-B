# Haze4K v2.30 NoPost ILFRB-ACS Compatibility-Gated OOF Table Policy Evidence

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy.md`

Status: `P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE`

Runtime server: `convir-4090`
Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`
Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Hard blocks:

- `training_launched: false`
- `p2b_selector_probe_launched: false`
- `locked_test_touched: false`

## Key Results

- Decision: `P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE`
- Safe-set restricted oracle mean/hard/easy: `0.6748830795288085 / 1.2371026992797851 / 0.1552295684814453`
- GT-free table policy mean/hard/easy: `0.014119386672973633 / 0.05647754669189453 / 0.0`
- hard-to-easy cross severe: `0.0`
- cross-bucket unsafe: `0.0`
- overstrong 1.5 unsafe: `0.2867132867132867`
- selected-to-table mean gap: `0.6607636928558349`

## Diagnosis

The safe-set restricted oracle remains useful, especially on hard samples, but
the fold-out GT-free LCB table policy is too weak to deploy. The route therefore
pauses at P2A: continue improving compatibility features/ranking/no-op
thresholding before any P2B selector probe, selector training, model training,
or locked-test evaluation.

Engineering retries before the final completed audit:

- `912fefa`: stopped before P2A due to missing argparse defaults.
- `c35cf34`: paused during read-only feature separability because the initial
  AUROC/threshold scan was quadratic on the replay table.
- `c794602`: optimized the read-only metrics and completed the audit.

## Primary Files

- `v230_p0_arch_contract_delta.md`
- `v230_p2a_two_phase_negative_control_report.csv`
- `v230_p2a_safe_set_restricted_oracle_gap.md`
- `v230_p2a_compatibility_feature_separability.csv`
- `v230_p2a_cross_bucket_firewall_ablation.csv`
- `v230_p2a_strength_dose_response_by_compat_bin.csv`
- `v230_p2a_lcb_constrained_oof_table_policy_report.csv`
- `v230_p2a_policy_action_confusion_matrix.csv`
- `v230_p2a_fold_tail_report.csv`
- `v230_p2a_closeout.json`
- `run_v230_p2a.sh`
- `monitor_v230.sh`
- `status.txt`

This directory is compact text evidence only. It excludes checkpoints, weights, images, arrays, archives, and raw feature dumps.
