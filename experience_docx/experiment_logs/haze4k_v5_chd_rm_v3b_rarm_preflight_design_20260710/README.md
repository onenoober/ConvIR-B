# v3b RARM Preflight Design

Status: `COMPLETED_PREFLIGHT_BLOCKED`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3b-rarm-preflight-design.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3b is the written no-training preflight/design decision authorized by the v3a
D7c-gated no-op connection pass. It checks whether the current runnable
training/evaluation entrypoints can legally proceed toward RARM/training with
`fam2_d7c_noop`.

## Evidence Files

- `v3b_static_preflight_source_audit.json`
- `v3b_rarm_preflight_closeout.json`
- `status.txt`
- `experience_docx/tools/preflight_chd_rm_v3b_rarm_design_static.py`

## Key Findings

- v3a passed only the no-training connection audit:
  `V3A_D7C_GATED_NOOP_CONNECTION_PASS_AUTHORIZE_NO_TRAINING_RARM_PREFLIGHT_ONLY`.
- `fam2_d7c_noop` requires an external `d7c_gate`.
- Current train/valid/eval entrypoints do not compute or pass `d7c_gate`.
- The train-time modulation-stat path also does not pass `d7c_gate`.
- The existing cloud v3a workspace is dirty and behind the GitHub v3a pass
  head, so it is not a clean direct parent runtime workspace.

## Decision

`V3B_RARM_PREFLIGHT_BLOCKED_GATE_PIPELINE_ABSENT_NO_RARM_TRAINING`

Do not run RARM/training/adapter work from v3b. The next route must first design
and audit a gate-producing forward contract for train/valid/eval and modulation
diagnostics.
