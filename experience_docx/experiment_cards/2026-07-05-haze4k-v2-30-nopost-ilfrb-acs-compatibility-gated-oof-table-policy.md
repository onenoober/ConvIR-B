# Haze4K v2.30 NoPost ILFRB-ACS Compatibility-Gated OOF Table Policy

Date: 2026-07-05

Branch: `codex/haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`

Route id: `haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705`

Status: `P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE`

## Hypothesis

v2.29 was a correct P2A pause, but it refines the bottleneck: the OOF action
bank still contains useful and tail-safe selected actions, especially for hard
samples, while the current GT-free table policy cannot yet identify compatible
source bucket, stage, and strength per target sample. v2.30 tests whether a
compatibility gate, hard-to-easy firewall, local dose-response analysis, and an
LCB-risk constrained OOF table can reduce cross-bucket and overstrong risk
without using per-sample target dPSNR for deployment policy selection.

## Parent And Architecture Contract

- Parent route: `codex/haze4k-v2-29-nopost-ilfrb-acs-safe-oof-action-bank-calibration`
- Parent commit: `936e3e0`
- Model architecture delta versus v2.29: none.
- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output, RGB output-output residual, or learned RGB
  post-output correction is introduced.
- P2B selector probing is blocked for this route.
- Training is blocked for this route.
- Locked Haze4K test is blocked.

## P2A Audit Scope

The audit reuses the v2.29 train-derived OOF replay and adds:

- two-phase negative-control interpretation;
- safe-set restricted oracle gap;
- compatibility feature separability;
- hard-to-easy firewall ablation;
- local strength dose-response by compatibility bin;
- LCB-risk constrained fold-out table policy;
- policy action confusion versus the safe-set restricted oracle.

The two baselines are:

- signal baseline: `bucket_strength_grid`;
- safety baseline: `energy_norm_plus_bucket_strength_plus_alignment_gate`.

The hybrid policy is `v230_hybrid_compat_firewall_lcb`.

## P2A Gates

GT-free table policy must pass:

- table-policy mean at least `+0.30`;
- table-policy hard at least `+0.60`;
- table-policy easy at least `0.00`;
- table-policy p05 at least `-0.15`;
- table-policy CVaR5 at least `-0.35`;
- table-policy severe rate at most `0.035`;
- table-policy fold-tail pass at least `4/5`;
- hard-to-easy cross severe rate at most `0.35`, preferred `<=0.10`;
- cross-bucket unsafe rate at most `0.35`, preferred `<=0.30`.

If the safe-set restricted oracle remains high but the table policy is low,
the bottleneck is GT-free policy/features, not the bank. If both are low, the
compatibility gate or bank action space is too restrictive. If hard-to-easy
severe remains high, the source-target compatibility mechanism is not solved.

## Final P2A Result

Runtime server: `convir-4090`

Final audited commit: `c794602`

Completion time: `2026-07-05T16:01:27+08:00`

Decision: `P2A_FAIL_COMPATIBILITY_GATED_TABLE_POLICY_PAUSE`

Key metrics:

- safe-set restricted oracle mean/hard/easy: `0.674883 / 1.237103 / 0.155230`
- GT-free table policy mean/hard/easy: `0.014119 / 0.056478 / 0.000000`
- table policy p05/CVaR5/severe/fold-tail: `-0.050298 / -0.305568 / 0.037500 / 4/5`
- hard-to-easy cross severe: `0.000000`
- cross-bucket unsafe: `0.000000`
- overstrong 1.5 unsafe: `0.286713`
- safe-set restricted oracle to table mean gap: `0.660764`

Interpretation:

The compatibility firewall removed the hard-to-easy and cross-bucket accepted
risk in the strict diagnostic set, and the safe-set restricted oracle still has
meaningful OOF gain, especially on hard samples. The fold-out GT-free LCB table
does not capture that signal: mean and hard gain are far below gate, and severe
rate is slightly above the `0.035` cap. This is a normal P2A scientific pause:
the remaining bottleneck is GT-free compatibility features/ranking/no-op
thresholding, not a reason to launch P2B, selector training, model training, or
locked-test evaluation.

Engineering notes:

- `912fefa` stopped before P2A due to missing argparse defaults inherited from
  the v2.27/v2.29 helpers.
- `c35cf34` was paused during feature separability because the first
  AUROC/threshold implementation was quadratic on the replay table.
- `c794602` optimized those read-only metrics and completed the P2A audit.

## Evidence

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_30_nopost_ilfrb_acs_compatibility_gated_oof_table_policy_20260705/`

Compact text artifacts:

- `README.md`
- `status.txt`
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

## Locked-Test Policy

Locked test is blocked. This route uses only train-derived Haze4K samples and
does not launch P2B selector probing, selector training, model training, or
deployment evaluation.
