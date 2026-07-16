# Generic Run Monitoring Validation

Date: 2026-07-16

Status: `RUNNING`

## Identity

- Route id: `generic_run_monitoring_validation_20260716`.
- Question: Can metadata-only monitoring provide bounded liveness and later
  closeout recovery without model polling or workload interference?
- Rules commit: `39009822a54aa0e756ab196d1be2d0d7aa0fae7c`.
- Source branch/commit: GitHub
  `main@39009822a54aa0e756ab196d1be2d0d7aa0fae7c`.
- Route branch: `codex/generic-run-monitoring-20260716`.
- Locked test/canary policy: prohibited; this validation uses no dataset, model,
  checkpoint, GPU, inference, training, evaluation, canary, or locked test.

## Scientific Contract

- Population and analysis/grouping unit: synthetic process/file-state cases.
- Intervention or factor contrast and reference: telemetry enabled versus the
  same synthetic parent workload; code audit excludes control/data access.
- Primary outcome, direction and aggregation: all fixed binary safety/recovery
  cases pass and 100 atomic pulses remain below the fixed low-cost bound.
- Preferred mechanism and strongest competing explanation: an external
  metadata-only writer provides liveness; competing risk is that it controls or
  blocks the workload, leaks scientific data, or makes a stale receipt terminal.
- Evidence roles and candidate/freeze point: `engineering_debug` only; code,
  tests, thresholds, and cases freeze in the route commit.
- Primary gate, uncertainty and threshold source: deterministic integrity gate;
  100 pulses, CPU time `<5s`, exactly one heartbeat file, no forbidden control
  token, and all unit tests pass.
- `PASS` authorizes: main-branch integration review for generic future use.
- `INCONCLUSIVE` authorizes: none; fix one engineering root cause on a new
  output identity.
- `FAIL` stops: adoption and use in any experiment runner.

## Implementation Contract

- Exact change and disabled mechanisms: add fail-open telemetry, bounded stale
  recovery, protocol, and tests; no experiment/model/data changes.
- Checkpoint/load/init/freeze contract: not applicable.
- Input whitelist and prohibited inputs: route/run/phase/progress/PID/path only;
  prohibit metrics, results, images, arrays, checkpoints, datasets, logs, and
  GPU state.
- Dataset/split/preprocessing/metric identities: no dataset.
- Matched baseline and budget: synthetic workload must finish normally; one CPU
  process, no GPU, hard runner timeout 10 minutes.
- Resource/cost limits or descriptive-only rationale: 100 pulses CPU `<5s`;
  heartbeat payload `<4096` bytes; no persistent process after its parent exits.
- Runner and required assets: `experience_docx/tools/run_generic_run_monitoring_validation.sh`;
  tracked Python sources and tests only, with no external runtime asset.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `GENERIC_MONITOR_VALIDATION` | engineering-only synthetic cloud validation | all tests and cost/control checks pass | main integration review only |

- First operation: GENERIC_MONITOR_VALIDATION
- Expected wall time and monitor profile: under 10 minutes; `short`.
- Complete-unit resume policy: none; new output only.
- Cloud workspace/run/output/status/closeout: derived by schema-v4 MCP.
- Compact Git evidence and cloud-only raw artifacts: summary and typed closeout
  are compact; test stdout remains in the cloud output.

## Prelaunch Control-Plane Incident

- The first planned output identity `generic-monitor-validation-r1` did not
  reach SSH: the MCP's internal temporary script conflicted with the committed-
  script-only transport contract. It produced no receipt and no runner.
- One committed read-only inspection at route commit
  `862dc4533d8ce351937204cb849b6412f3c303f8` confirmed that the exact remote
  repository, output path, closeout, and tmux session were all absent.
- The single allowed repair keeps the six-tool schema and fixed host, sends only
  internally generated bodies through bounded SSH stdin, and changes the next
  output identity to `generic-monitor-validation-r2`. No experiment scope,
  model, data, gate, or authorization changed.

## Decision

- Verdict and primary reason: isolated candidate gate passed; six telemetry and
  22 control-plane tests passed with a typed `COMPLETED_GATE_PASS` closeout.
- Mechanism/control and safety reason: telemetry remained metadata-only and
  fail-open; semantic audit found no process/GPU control or scientific-output
  read, and control-plane observation remained bounded and receipt-bound.
- Evidence-independence and cost reason: no model, GPU, dataset, checkpoint,
  canary, or locked test was used; 100 pulses consumed `0.160066823` CPU
  seconds and produced only one 230-byte heartbeat file.
- Authorized next action or terminal stop: main-integration review only,
  followed by registered-service schema-v4 end-to-end validation. Model
  experiments and project-default adoption remain blocked until that passes.

## Candidate Validation History

- `candidate-7b1ee1580-r1` ended `FAILED_ENGINEERING` before the control-plane
  and cost phases. Five telemetry behavior tests passed; the sixth raw-text
  source audit falsely matched `signal.` in explanatory prose.
- The typed closeout records model calls `0` and GPU, dataset, checkpoint,
  canary, and locked-test access all `false`. No model experiment ran.
- This authorizes exactly one semantic source-audit correction and a fresh
  candidate identity. It does not authorize adoption, `main` integration, MCP
  reload, or any model operation.
- `candidate-4bbd03154-r2` also ended `FAILED_ENGINEERING` before control-plane
  and cost phases. The new syntax-tree audit skipped the legitimate
  `/proc/PID/stat` read because the method receiver was a compound path
  expression; five telemetry tests again passed.
- Its typed closeout again records model calls `0` and every protected-resource
  flag `false`. This authorizes only one compound-receiver matcher correction
  and a fresh candidate identity; all adoption and model operations remain
  blocked.
- `candidate-33a772268-r3` completed all three milestones. Telemetry tests were
  6/6, control-plane tests were 22/22, semantic audit findings were empty, the
  CPU/file gate passed, and its closeout authorizes
  `MAIN_INTEGRATION_REVIEW_ONLY`. This is not yet permission to run a model
  experiment.
