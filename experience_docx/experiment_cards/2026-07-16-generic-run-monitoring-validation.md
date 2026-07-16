# Generic Run Monitoring Validation

Date: 2026-07-16

Status: `COMPLETED`

## Identity

- Route id: `generic_run_monitoring_validation_20260716`.
- Question: Can metadata-only monitoring provide bounded liveness and later
  closeout recovery without model polling or workload interference?
- Rules commit: `dca94d71c9fe73e4e93910b0587927c79ab7023c`.
- Source branch/commit: GitHub
  `main@dca94d71c9fe73e4e93910b0587927c79ab7023c`.
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
- `PASS` authorizes: candidate pass authorizes main integration review; the
  later receipt-bound E2E pass authorizes generic monitoring adoption only.
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
- Runner and required assets: `experience_docx/tools/run_generic_run_monitoring_validation.sh` and
  `experience_docx/tools/run_generic_run_monitoring_e2e_validation.sh`;
  tracked Python sources and tests only, with no external runtime asset.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `GENERIC_MONITOR_VALIDATION` | engineering-only synthetic cloud validation | all tests and cost/control checks pass | main integration review only |
| `MAIN_INTEGRATION_REVIEW_ONLY` | receipt-bound registered-service E2E | server 4.1.0, six schema-v4 tools, tests/cost, closeout and evidence fetch pass | generic monitoring adoption only |

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

- Verdict and primary reason: adopted; the strengthened candidate and final
  receipt-bound registered-service E2E both passed their frozen gates.
- Mechanism/control and safety reason: telemetry remained metadata-only and
  fail-open; semantic audit found no process/GPU control or scientific-output
  read, and control-plane observation remained bounded and receipt-bound.
- Evidence-independence and cost reason: no model, GPU, dataset, checkpoint,
  canary, or locked test was used; final E2E 100-pulse CPU was
  `0.154378791` seconds and produced only one 231-byte heartbeat file.
- Authorized next action or terminal stop: adopt generic monitoring for future
  long cloud operations. This closeout does not authorize A1X, training,
  evaluation, inference, canary, or locked-test access.

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
- `candidate-48203394e-r4` repeated the entire gate after adding a seventh
  lifecycle test. The unlimited sidecar exited after the exact parent was
  naturally reaped, all seven telemetry and 22 control-plane tests passed, and
  the typed closeout again authorizes only main-integration review.

## Registered-Service End-to-End Gate

- GitHub main integration completed by non-force fast-forward at
  `dca94d71c9fe73e4e93910b0587927c79ab7023c`; the dedicated
  `ConvIR-B-operations-v4` worktree is clean at that exact commit.
- The final CPU-only schema-v4 operation uses the fresh output identity
  `generic-monitor-e2e-r1`. It must verify server `4.1.0`, exactly six tools,
  plan/start/receipt/finish, typed closeout, and receipt-bound evidence fetch.
- The current Codex MCP transport did not reconnect after its uniquely
  identified old process was normally terminated. Therefore the same
  registered executable and path will be exercised through one isolated stdio
  instance and isolated state directory. This does not broaden its six-tool
  interface or experiment scope.
- The isolated registered-path instance reported server `4.1.0`, source SHA-256
  `dc07ac60056b5e7da52f045419f196c82f6a19a64255408504056b1329ecc2ae`,
  and exactly six schema-v4 tools. It completed `PLAN_READY -> LAUNCHED ->
  CLOSEOUT_VALIDATED`, fetched the two receipt-bound evidence files without Git
  mutation, and validated the adoption terminal tuple.
- The final closeout SHA-256 is
  `2e6f627eb5b3a6b0d3645aa0cfd91b43992d2625a8df41d345a77270cbad8f72`;
  summary SHA-256 is
  `5199e564a6da93a77ffcd8178c6415ccfb4511ae299cd79d6430087959eddb2c`.
