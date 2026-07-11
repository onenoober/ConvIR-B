# v3m Blockwise Counterfactual Advantage

Status: `PREFLIGHT_A0B_DENSE_CONTINUOUS_CROSS_AUDIT_ONLY`.

Route card:
`experience_docx/experiment_cards/2026-07-11-haze4k-v5-chd-rm-v3m-blockwise-counterfactual-advantage.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

This route starts from the v3l frozen direct operators and rechecks control
granularity under a common action ladder before any local actuation, physics,
proxy, or controller work. It is train-derived only and keeps locked test and
canary blocked.

## A0a Outputs

- `v3m_a0_common_action_summary.json`
- `v3m_a0_common_action_gate.csv`
- `v3m_a0_granularity_retention_group_bootstrap.csv`
- `v3m_a0_common_action_oracle_summary.csv`
- `v3m_a0_confirm_audit_only_policy_summary.csv`
- `v3m_a0_operator_agreement.csv`
- `v3m_a0_source_manifest.json`

Cloud-only per-image outputs live under `cloud_only_raw_common_action/` and
must not be committed or synced to GitHub.

`v3m_deviation_log.md` records the initial summary-only engineering failure and
the constrained rebuild command.

## A0a Closeout

`v3m_a0a_closeout.md` is the canonical compact A0a result and integrity
record. It records the dual-operator gate values, the raw-table preservation
checks, and the repaired finite operator-agreement statistics.

Decision:
`V3M_A0_COMMON_ACTION_GRANULARITY_PASS_AUTHORIZE_A0B_DENSE_AND_CONTINUOUS_MECHANISM_ONLY`.

Only the A0b dense-grid and continuous-pixel mechanism cross-audit is
authorized. `v3m_a0b_metric_contract.md` freezes its paired OOF source hashes,
policy mappings, and quantization-gap gate before it reads any paired outcome.
No model/inference rerun is needed. It cannot promote a policy, controller,
training, canary, or locked-test access; only its own written pass closeout may
authorize the A1 feasible-local-actuation audit.
