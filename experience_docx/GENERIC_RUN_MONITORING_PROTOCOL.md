# Generic Run Monitoring Protocol

Date: 2026-07-27

Status: adopted for future long cloud operations. The reviewed generic files
are on GitHub `main@dca94d71c9fe73e4e93910b0587927c79ab7023c`, and the
receipt-bound registered-service E2E gate passed at route commit
`c84823d2c135936c0793768eecaa654dc206ca2f`. Adoption does not authorize any
model experiment or change a route's scientific gate.

## Non-Interference Boundary

Long operations run without a resident model or polling watcher. Monitoring is
metadata-only. It cannot read scientific metrics, predictions, images, arrays,
checkpoints, datasets, GPU state, runtime logs, or model state. It cannot
signal, stop, pause, resume, restart, or relaunch the workload.

The workload owns scientific progress and its typed closeout. Monitoring owns
only liveness metadata:

| File | Writer | Contract |
| --- | --- | --- |
| `OUTPUT_PATH/heartbeat.json` | optional fail-open sidecar | atomically replaced; identity, phase, progress counters, sequence, parent identity, and time only |
| `OUTPUT_PATH/status.txt` | workload runner | append-only milestone events; no scientific result or interim gate |
| route evidence `*_closeout.json` | workload runner | the only terminal state, decision, and authorization evidence |

If telemetry cannot write, it prints one `RUN_TELEMETRY_DEGRADED` line and
exits zero. The parent workload continues. Hard wall-time, complete-unit state,
resume, failure classification, evidence retention, and closeout remain runner
responsibilities.

## Runner Integration

Use the explicit cloud Python and tracked generic helper. Start the sidecar only
after identity/output preflight and before expensive work:

```bash
TELEMETRY="$REMOTE_REPO/experience_docx/tools/run_telemetry.py"
HEARTBEAT_PATH="$OUTPUT_PATH/heartbeat.json"
STATUS_PATH="$OUTPUT_PATH/status.txt"

"$PY" "$TELEMETRY" sidecar \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase workload \
  --heartbeat "$HEARTBEAT_PATH" --parent-pid "$$" \
  --interval-seconds 60 &

"$PY" "$TELEMETRY" event \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase preflight \
  --status "$STATUS_PATH" --event preflight_pass \
  --completed 0 --total "$TOTAL_UNITS"
```

At each complete resumable unit, atomically seal the unit state/hash first and
then append one `unit_complete` event. Do not publish a scientific metric in
the event. A direct `pulse` may update phase/progress without creating another
long-lived process.

Do not wait on, kill, or use the telemetry PID to control the run. The sidecar
observes the original parent PID plus Linux start ticks and exits by itself when
that exact process disappears.

## Observation Policy

`convir_route_finish` prefers `heartbeat.json`, falls back to `status.txt`, and
then to receipt launch time. One call observes a bounded 30- or 60-second
window.

- Healthy active sealed finish: return the frozen running state and normally
  finish near the expected end.
- Operator progress: `progress_only` may refresh before ETA with a separate
  15-second minimum interval and finite budget. Return only stage, completed/
  total units, exact-session activity, heartbeat age/source, snapshot time and
  cached/current identity. Never expose outcomes, metrics or data ids.
- Early terminal: the progress probe reports `TERMINAL_DETECTED`, clears the
  sealed not-before cache and permits immediate closeout validation without
  revealing the tuple in the probe.
- Stale active: return `MONITOR_STALE` as monitoring/infrastructure evidence.
  The receipt remains open so a later typed closeout can still be validated.
  Do not create a watcher; the operator may later request another bounded
  result-blind snapshot or perform the single allowed engineering review.
- Dead without closeout: close as `CLOSEOUT_MISSING` and allow one engineering
  review.
- Typed closeout: validate route/run/commit/runner identity, allowed terminal
  tuple, and SHA-256, then close the receipt.
- Operator cancel: validate receipt-derived route/run/commit/runner/repo/output/
  session and exact process owner/environment/command/start ticks; write the
  request, terminate gracefully, then revalidate before bounded escalation.
  Never accept a PID input. Close as
  `CANCELLED_BY_OPERATOR / null / NONE` and keep evidence locked.

Staleness never changes the scientific decision, triggers relaunch, or by
itself authorizes termination. The receipt retains 64 sealed finish windows
plus a separate finite progress-refresh budget as hard misuse guards. Explicit
human cancellation is an independent control action and never a scientific
decision.

## Default Timing

| Operation | Heartbeat interval | Stale threshold | Model-visible checks |
| --- | ---: | ---: | --- |
| smoke/preflight | 30-60 seconds | at least 3 intervals | launch plus sealed finish; result-blind refresh on operator request |
| long train/eval/formal | 60 seconds | 300 seconds | launch plus sealed finish; bounded result-blind refresh or cancel on operator request |

An operation may choose a longer interval when individual kernels routinely
block CPU scheduling, but must keep the stale threshold at least three times
the interval. Monitoring cadence is never a scientific factor.

## Adoption Evidence

Cloud synthetic validation passed telemetry atomicity,
identity binding, fail-open unwritable-path behavior, parent non-interference,
heartbeat/status/launch fallback, stale-then-closeout recovery,
dead-without-closeout finite stop, receipt-bound closeout, and the fixed CPU/file
cost gate. A syntax-tree safety audit must reject actual process/GPU control and
non-`/proc` reads while ignoring explanatory prose. No model, dataset,
checkpoint, GPU, or scientific result is used.

The strengthened candidate passed seven telemetry tests and 22 restricted
control-plane tests. The final E2E used the registered `convir-ops 4.1.0`
executable, exactly six schema-v4 tools, a fresh plan/output/receipt, one bounded
finish window, and receipt-bound evidence fetch. Its terminal tuple is
`COMPLETED_GATE_PASS / GENERIC_RUN_MONITORING_E2E_PASS /
GENERIC_RUN_MONITORING_ADOPTION`. One hundred pulses consumed
`0.154378791` CPU seconds, projected to `0.0926272746` CPU seconds per hour at
a 60-second interval, and created only one 231-byte `heartbeat.json`.
