# v3m Blockwise Counterfactual Advantage

Status: `A1_SMOKE_PASS_FORMAL_OOF_ONLY`.

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

The initial A0b output is retained as a metric-contract-mismatched diagnostic:
continuous alpha is solved before final output clamping, so a pointwise nested
candidate check is invalid for that pair. `v3m_a0b_metric_contract.md` records
the correction and A0b-r1 remains read-only.

## A0b-r1 Closeout

`v3m_a0b_r1_closeout.md` records the valid corrected result. All eight paired
quantization-gap checks passed with exact fixed-alpha replay, input SHA checks,
and existing tail-safety preservation. Decision:
`V3M_A0B_QUANTIZATION_GAP_SMALL_AUTHORIZE_A1_FEASIBLE_LOCAL_ACTUATION_ONLY`.

Only A1 feasible-local-actuation audit is now authorized. It needs a separate
metric contract and must not train a controller, select route-confirm choices,
launch a canary, or access locked test.

## A1 Preflight

`v3m_a1_metric_contract.md` freezes a no-training block16 local-observability
audit. It writes its large block table only on cloud and tests four fixed
signals with grouped per-image OOF AUC. `run_v3m_a1_smoke32.sh` is the required
32-image replay template; the first attempt is retained as an engineering
failure, r1 preserved the wrong subset fold map, and
`run_v3m_a1_smoke32_r2.sh` is the corrected screen. Only its
explicit pass marker authorizes
`run_v3m_a1_formal.sh` on all 1,200 OOF images.

## A1 Smoke Result

The corrected r2 smoke reproduced both frozen operators' 32 fixed-alpha rows
exactly and wrote the expected 40,000 cloud-only block records. Its compact
summary, replay CSV, signal CSV, and source manifest are retained under
`a1_smoke32_r2/`; the raw block table is cloud-only. Decision:
`V3M_A1_SMOKE_REPLAY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`.
