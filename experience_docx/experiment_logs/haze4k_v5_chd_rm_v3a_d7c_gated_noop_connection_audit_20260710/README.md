# v3a D7c-Gated No-Op Connection Audit

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3a-d7c-gated-noop-connection-audit.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3a verifies the connection risk between the validated D7c deployable prior and
the FAM2 no-op modulation shell. D7c gate tensors may enter the candidate
forward path, but final modulation must remain zero-init no-op and exact
A0-equivalent.

## Authorized Scope

- no training;
- no RARM;
- no adapter;
- no loss changes;
- no ConvIR-B unfreeze;
- no locked Haze4K test;
- train-derived internal val-inner 600 only.

## Expected Evidence

- `d7c_noop_state_dict_compatibility.json`
- `d7c_noop_modulation_zero_stats.json`
- `d7c_noop_random_equivalence.json`
- `d7c_noop_real_batch_equivalence.json`
- `d7c_noop_internal_val600_summary.json`
- `d7c_noop_per_image_diff_summary.csv`
- `d7c_noop_closeout.json`
- `forbidden_flow_audit.json`
- `v3a_d7c_gated_noop_connection.log`
- `status.txt`

## Metric Contract

Pass only if D7c gate tensors are nontrivial on real/internal samples and all
A0-vs-candidate output/metric deltas remain numerically zero within the written
thresholds.

## Decision

Pending.

## Attempt Notes

- Attempt 1 was an engineering no-op-expression failure: internal val-inner 600
  passed with exact `0.0` output and metric deltas, but the real batch gate
  exceeded the stricter `1e-7` output threshold with max diff `3.576e-7`.
  Cause: `fused * (1 + gamma)` can perturb CUDA outputs even when `gamma == 0`.
  Correction: use residual identity form `fused + fused * gamma + beta`.
