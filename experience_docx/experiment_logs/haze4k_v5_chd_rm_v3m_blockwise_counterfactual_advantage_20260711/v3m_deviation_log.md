# v3m Deviation Log

## 2026-07-11 A0a summary failure

The original `run_v3m_a0_common_action_oracle.sh` invocation completed all
frozen D_ref/D_rep OOF and confirm-audit replay segments. It then failed in
`percentile()` during grouped bootstrap summary construction because NumPy arrays
cannot be used as scalar booleans.

Verified retained raw outputs:

- policy rows: 43,200 rows plus header;
- OOF policy rows: 28,800 rows plus header;
- direction rows: 3,600 rows plus header;
- block rows: 7,200 rows plus header.

The recovery command is restricted to `--summarize_existing_raw`. It does not
load models or checkpoints, rerun inference, modify raw rows, change the action
set, or access locked test. The original failed launcher remains recorded in
`status.txt` and its log.
