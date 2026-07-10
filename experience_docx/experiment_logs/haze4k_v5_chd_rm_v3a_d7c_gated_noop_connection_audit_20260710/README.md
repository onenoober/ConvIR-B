# v3a D7c-Gated No-Op Connection Audit

Status: `COMPLETED_GATE_PASS`

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

`V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY`

v3a passes as a no-training/no-RARM connection audit. Real D7c gate tensors are
connected into the FAM2 no-op shell, the gates are nontrivial on internal
samples, and the final candidate remains exact A0-equivalent on the declared
checks.

Key final evidence from attempt 5:

- state compatibility passed with only `FAM2.modulator.weight` and
  `FAM2.modulator.bias` missing from the A0 checkpoint;
- parameter delta is exactly `8320`;
- internal val-inner 600 output max absolute diff is `0.0`;
- internal val-inner 600 PSNR and SSIM delta max absolute values are `0.0`;
- nontrivial D7c gate coverage is `599/600`;
- `d7c_noop_per_image_diff_summary.csv` has `600` rows;
- `forbidden_flow_audit.json` records no locked test, no training, no RARM, no
  adapter training, and no ConvIR-B unfreeze.

Next authorized stage remains conservative: RARM/training is still blocked.
Only a separately written preflight/design decision may be opened after this
audit.

## Attempt Notes

- Attempt 1 was an engineering no-op-expression failure: internal val-inner 600
  passed with exact `0.0` output and metric deltas, but the real batch gate
  exceeded the stricter `1e-7` output threshold with max diff `3.576e-7`.
  Cause: `fused * (1 + gamma)` can perturb CUDA outputs even when `gamma == 0`.
  Correction: use residual identity form `fused + fused * gamma + beta`.
- Attempt 2 still showed a batch-mode real-batch diff of `2.384e-7` while
  val-inner 600 remained exactly `0.0`. Correction before attempt 3: force
  deterministic cuDNN and disable TF32 for the audit process.
- Attempt 3 passed random, real batch, and val-inner 600 equivalence, but
  closeout still failed because the audit script retained the obsolete
  `[128,65,1,1]` expected modulator shape from the earlier gate-concat design.
  Correction before attempt 4: expected shape is `[128,64,1,1]` because D7c gate
  externally multiplies gamma/beta and adds no parameters.
- Attempt 4 passed random, real batch, and val-inner 600 equivalence, but
  closeout still failed because the audit compared sorted candidate missing keys
  against an unsorted expected list. This was an audit key-order bug, not a
  model or metric failure.
- Attempt 5 reran the same cloud audit after the key-order fix and passed with
  `V3A_D7C_GATED_NOOP_CONNECTION_OK`.
