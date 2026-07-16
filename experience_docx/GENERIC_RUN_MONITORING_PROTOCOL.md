# Generic Run Monitoring Protocol

Date: 2026-07-16

Status: validation candidate. It becomes the default only after the synthetic
cloud closeout passes and the reviewed generic files reach GitHub `main`.

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

- Healthy active: return `MONITOR_OBSERVED`; inspect again only near the frozen
  expected end.
- Stale active: return `MONITOR_STALE` as monitoring/infrastructure evidence.
  The receipt remains open so a later typed closeout can still be validated.
  Do not poll repeatedly; wait until the expected end or perform the single
  allowed engineering review.
- Dead without closeout: close as `CLOSEOUT_MISSING` and allow one engineering
  review.
- Typed closeout: validate route/run/commit/runner identity, allowed terminal
  tuple, and SHA-256, then close the receipt.

Staleness never changes the scientific decision, triggers relaunch, or
authorizes termination. The receipt retains the finite maximum of 64 total
observation windows as a hard misuse guard.

## Default Timing

| Operation | Heartbeat interval | Stale threshold | Model-visible checks |
| --- | ---: | ---: | --- |
| smoke/preflight | 30-60 seconds | at least 3 intervals | launch plus one expected-end check |
| long train/eval/formal | 60 seconds | 300 seconds | launch plus one expected-end check; one later check only if still healthy |

An operation may choose a longer interval when individual kernels routinely
block CPU scheduling, but must keep the stale threshold at least three times
the interval. Monitoring cadence is never a scientific factor.

## Adoption Gate

Before adoption, cloud synthetic validation must pass telemetry atomicity,
identity binding, fail-open unwritable-path behavior, parent non-interference,
heartbeat/status/launch fallback, stale-then-closeout recovery,
dead-without-closeout finite stop, receipt-bound closeout, and the fixed CPU/file
cost gate. A syntax-tree safety audit must reject actual process/GPU control and
non-`/proc` reads while ignoring explanatory prose. No model, dataset,
checkpoint, GPU, or scientific result is used.
