# Haze4K CHD-RM v3b RARM Preflight Design

Date: 2026-07-10

Status: `COMPLETED_PREFLIGHT_BLOCKED`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3b_rarm_preflight_design_20260710/`

Parent route branch:
`github/codex/haze4k-v5-v3a-d7c-gated-noop-connection-audit`

Route branch:
`codex/haze4k-v5-v3b-rarm-preflight-design`

## Route Identity

v3b is the separate no-training preflight/design decision authorized by v3a. It
does not implement RARM and does not run training, adapter work, canary
expansion, evaluation, or locked-test access.

The question is narrower: can the current runnable training/evaluation
entrypoints safely support the v3a `fam2_d7c_noop` candidate as the next step
without a new gate-producing runtime design?

## Fact Sources

- GitHub `main` at `7c3fbddee080f8c5adf305e433c88b259fcd340e`:
  `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.
- GitHub v3a route branch at
  `ef3310d8ec47375141d9ab7d62d5d320122e18a1`.
- v3a closeout:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3a_d7c_gated_noop_connection_audit_20260710/d7c_noop_closeout.json`.
- v3b source inspection in this branch:
  `Dehazing/ITS/models/ConvIR.py`, `Dehazing/ITS/train.py`,
  `Dehazing/ITS/valid.py`, and `Dehazing/ITS/eval.py`.
- Cloud state audit on `convir-4090` for the v3a workspace:
  cloud head `58a356e`, `41` status lines, therefore not a clean parent
  runtime workspace for v3b.

## Authorized Scope

- static/source preflight only;
- no local runtime validation;
- no cloud training/evaluation launch;
- no RARM connection or RARM training;
- no adapter training;
- no ConvIR-B unfreeze;
- no loss change;
- no canary expansion;
- no locked Haze4K test.

## Metric Contract

Pass only if the current runnable entrypoints already have a legal path to
produce and pass a nontrivial D7c gate into `fam2_d7c_noop` for training,
validation, evaluation, and modulation statistics without expanding into a
forbidden flow.

Fail if `fam2_d7c_noop` requires a `d7c_gate` but any required entrypoint still
calls the model or modulation-stat path without one.

## Preflight Result

The preflight fails before any RARM/training launch.

Source evidence:

- `Dehazing/ITS/models/ConvIR.py:68` raises
  `ValueError('d7c_gate is required for d7c_noop FAM mode')`.
- `Dehazing/ITS/models/ConvIR.py:86` applies the same requirement for
  modulation stats.
- `Dehazing/ITS/train.py:107` still calls `pred_img = model(input_img)`.
- `Dehazing/ITS/valid.py:32` still calls `pred = model(input_img)[2]`.
- `Dehazing/ITS/eval.py:42` still calls `pred = model(input_img)[2]`.
- `Dehazing/ITS/train.py:28` still calls
  `model.collect_modulation_stats(input_img)`.

Therefore the current training/validation/evaluation path would either fail
when `fam2_d7c_noop` is selected or silently require a new, unapproved gate
pipeline. v3a proved only that an externally supplied D7c gate can be connected
without perturbing A0. It did not create a training/evaluation entrypoint that
computes and passes that gate.

## Cloud Preflight

The existing cloud v3a workspace is not a valid direct parent runtime workspace:

- path:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v3a-d7c-gated-noop-connection-audit`;
- cloud head: `58a356e`;
- GitHub v3a pass head: `ef3310d8`;
- `git status --short` line count: `41`.

Any future cloud run must start from a fresh workspace from the pushed v3b or a
new written design branch, not from the dirty cloud v3a directory.

## Decision

`V3B_RARM_PREFLIGHT_BLOCKED_GATE_PIPELINE_ABSENT_NO_RARM_TRAINING`

RARM/training remains blocked. The blocker is not a no-op equivalence failure
and not a v3a numerical failure. The blocker is the absent gate-producing
runtime pipeline for the entrypoints that would be used by training,
validation, evaluation, and modulation diagnostics.

The next useful route, if opened later, must first design and audit the D7c gate
producer/forward contract as an entrypoint-level no-training preflight. It must
not begin with RARM training.
