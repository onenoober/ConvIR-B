# Haze4K v2.31 NoPost Target-Only Action-Value Identifiability Audit

Date: 2026-07-05

Branch: `codex/haze4k-v2-31-nopost-action-value-identifiability-audit`

Route id: `haze4k_v2_31_nopost_action_value_identifiability_audit_20260705`

Status: `P2A_FAIL_ACTION_VALUE_IDENTIFIABILITY_CLOSE_CURRENT_BANK`

## Hypothesis

v2.30 showed that compatibility gating can remove accepted hard-to-easy and
cross-bucket risk while preserving a useful safe-set restricted oracle, but the
GT-free table policy could not identify useful safe actions. v2.31 tests the
actual bottleneck: whether target-only physics/frequency/internal/A0-diagnostic
features contain enough deployable information to rank safe-set action value.

This is a P2A identifiability audit, not a P2B selector probe and not model
training.

## Parent And Architecture Contract

- Parent route: `codex/haze4k-v2-30-nopost-ilfrb-acs-compatibility-gated-oof-table-policy`
- Parent commit: `8971902`
- Model architecture delta versus v2.30: none.
- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output, RGB output-output residual, or learned RGB
  post-output correction is introduced.
- P2B selector probing is blocked for this route.
- Training is blocked for this route.
- Locked Haze4K test is blocked.

## P2A Audit Scope

The audit reuses the v2.30 train-derived OOF action replay and adds:

- target-only action-value separability with physics/frequency/internal
  features;
- nested fold-out safe-set ranking upper bound;
- physics-cluster OOF prototype-bank diagnostic;
- no-op risk-coverage curve;
- real-vs-shuffled and random/no-op/oracle leakage controls;
- policy-vs-safe-set action confusion;
- local optimum audit.

## P2A Gates

Feature separability gate:

- combined physics/frequency `is_useful_gt_0p30` AUROC at least `0.70`;
- hard `is_useful_gt_0p30` AUROC at least `0.70`;
- easy `should_noop` AUROC at least `0.80`;
- useful AUROC fold std at most `0.07`.

Nested ranking gate:

- best nested OOF policy mean at least `+0.30`;
- hard at least `+0.60`;
- severe rate at most `0.035`;
- safe-set-to-policy mean gap at most `+0.30`;
- real-vs-shuffled gap must be positive enough to support action-value signal.

## Decision Rule

If feature separability and nested ranking both pass:

`P2A_PASS_ACTION_VALUE_IDENTIFIABILITY`

Then a later task may consider P2B selector-probe dry run, still without
training the main model or touching locked test.

If either hard gate fails:

`P2A_FAIL_ACTION_VALUE_IDENTIFIABILITY_CLOSE_CURRENT_BANK`

Then close the current discrete action-bank selector route and pivot toward a
separate NoPost bounded internal low-frequency correction-field route rather
than continuing table/firewall micro-tuning.

## Evidence

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_31_nopost_action_value_identifiability_audit_20260705/`

## Final P2A Result

Runtime server: `convir-4090`

Final audited commit: `46122af`

Completion time: `2026-07-05T17:40:56+08:00`

Decision: `P2A_FAIL_ACTION_VALUE_IDENTIFIABILITY_CLOSE_CURRENT_BANK`

Key metrics:

- combined physics/frequency `is_useful_gt_0p30` AUROC all/hard:
  `0.6854 / 0.5444`
- easy `should_noop` AUROC: `0.5934`
- useful AUROC fold std: `0.1059`
- best nested ranker: `kNN_nonparametric`
- best ranker mean/hard/easy: `+0.4234 / +0.8152 / +0.0995`
- best ranker p05/CVaR5/severe: `-0.4421 / -1.2582 / 0.1375`
- safe-set oracle mean/hard/easy: `+0.6749 / +1.2371 / +0.1553`
- best ranker safe-set-to-policy gap: `+0.2515`
- best physics-cluster ranker: `adjacent_cluster_only`, mean/hard
  `+0.3712 / +0.5635`

Interpretation:

The added target-only physics/frequency/internal/A0-diagnostic features improve
mean and hard ranking versus the v2.30 table, but they do not meet the
identifiability gates. The useful-action feature gate fails, hard useful AUROC
is weak, easy no-op recognition is weak, and the best nested ranker is not
tail-safe. The route therefore closes the current discrete action-bank selector
candidate as a training/P2B route. No P2B selector probe, model training, or
locked-test command was launched.

Expected compact text artifacts:

- `README.md`
- `status.txt`
- `v231_p0_arch_contract_delta.md`
- `v231_p2a_action_value_feature_separability.csv`
- `v231_p2a_safe_set_ranking_upper_bound.md`
- `v231_p2a_physics_cluster_oof_prototype_bank.csv`
- `v231_p2a_noop_risk_coverage_curve.csv`
- `v231_p2a_real_vs_shuffled_action_value_controls.csv`
- `v231_p2a_policy_vs_safe_set_confusion_matrix.csv`
- `v231_p2a_local_optimum_audit.md`
- `v231_p2a_closeout.json`
- `run_v231_p2a.sh`
- `monitor_v231.sh`

## Locked-Test Policy

Locked test is blocked. This route uses only train-derived Haze4K samples and
does not launch P2B selector probing, selector training, model training, or
deployment evaluation.
