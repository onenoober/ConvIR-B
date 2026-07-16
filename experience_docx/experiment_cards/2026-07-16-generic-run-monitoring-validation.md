# Generic Run Monitoring Validation

Date: 2026-07-16

Status: `PLANNED`

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

Fill only after terminal evidence.

- Verdict and primary reason:
- Mechanism/control and safety reason:
- Evidence-independence and cost reason:
- Authorized next action or terminal stop:
