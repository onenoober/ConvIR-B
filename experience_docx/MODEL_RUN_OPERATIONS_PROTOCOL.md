# Model Run Operations Protocol

Date: 2026-07-16

Status: per-launch cloud lifecycle. Route design belongs to the start
checklist; formal interpretation belongs to the Gate Policy.

## Required Order

```text
typed prior authorization -> dynamic preflight -> tracked runner with
integrated smoke -> bounded observation -> typed closeout -> compact evidence
```

The frozen route card plus operations manifest authorize the first stage. Every
later stage requires the exact previous closeout. A semantic-preserving
engineering repair reuses that authority; it never creates a new R3 document.

Local WSL remains syntax/compile-only. Runtime runs only on `convir-4090`
with the explicit cloud Python.

## Dynamic Preflight

Immediately before launch verify only changing facts:

- route branch HEAD and clean cloud workspace match the exact route commit;
- the manifest's canonical rule-bundle digest still matches current GitHub
  `main`, or a changed bundle received one explicit compatibility review;
- runner and required asset hashes match the route commit/asset manifest;
- the first-stage card authorization or prior closeout tuple matches;
- data role and locked-test policy permit the exact stage;
- the selected GPU satisfies the frozen resource floor immediately before
  launch;
- session and output are new, or an exact unit-boundary resume is frozen;
- status, heartbeat, log, closeout, and retained-state paths are explicit.

Never substitute a commit, split, asset, environment, threshold, output, or
stage silently. A prelaunch failure is engineering state.

## Runner Contract

Use one parameterized tracked runner for the route. It must:

- use `set -euo pipefail` and the explicit cloud Python;
- write only to the route `RUN_ROOT`, except the compact closeout in the
  route evidence directory;
- run route-specific identity, shape, input-whitelist, no-op, finite and tiny
  update checks before expensive work in the same process;
- append phase, progress, heartbeat, timing, and terminal markers to
  `status.txt`;
- capture combined stdout/stderr and return the underlying exit code;
- reject unauthorized locked-test access;
- write route id, run id, route commit, runner SHA-256 and one allowed terminal
  tuple to its closeout;
- retain only the states needed by the frozen analysis.

## Bounded Resume

Permit resume only at a complete unit named before launch, such as a fold,
seed, factor cell, or checkpoint boundary. The runner must hash completed units
and run only missing units. It may not expose intermediate confirmation results
or change model, data, epoch, threshold, order, or gate after resume.

An interruption before a complete unit restarts that unit. A changed scientific
contract requires a new route/run id. Resume never needs a new scientific
authorization when the frozen contract and completed-unit hashes still match.

## Monitoring

The MCP `finish` tool performs one server-side observation window. Use one
current task for all healthy observations; never create a task per poll.

| Profile | Maximum window |
| --- | ---: |
| `short` | 30 seconds |
| `standard` | 60 seconds |
| `long` | 60 seconds |

Report only state, active session/process, progress unit, latest primary metric
when available, last marker, heartbeat age, and phase timing. A stale heartbeat
or ended session without closeout enters one engineering review. Do not poll a
dead session repeatedly.

## Typed Closeout

Each stage writes one compact JSON containing at least:

```json
{
  "route_id": "route",
  "run_id": "run",
  "stage": "s0",
  "route_commit": "40-hex",
  "runner_sha256": "64-hex",
  "evidence_role": "engineering_debug",
  "state": "COMPLETED_GATE_PASS",
  "decision": "PASS",
  "authorizes": "formal",
  "reason": "compact evidence-backed reason"
}
```

Use `decision: null` for command, infrastructure, or engineering-invalid
runs. The MCP validates provenance and the allowed tuple; it never interprets a
scientific result.

Minimum compact evidence is the runner, status excerpt, closeout, evidence
README, primary aggregate summary, and required state hashes/counts. Raw logs,
checkpoints, images, arrays, predictions, and large tables stay in
`RUN_ROOT`.

## Finite Failure Handling

| Failure | Action |
| --- | --- |
| quoting/CRLF/PATH/marker | label `FAILED_COMMAND`; apply one canonical transport correction |
| missing/mismatched asset or import | one engineering repair cycle under the same scientific contract |
| resource unavailable before launch | return `RESOURCE_WAIT_REQUIRED`; later retry the exact plan without changing thresholds |
| timeout after launch boundary | return `START_STATE_UNKNOWN`; inspect once before any action |
| NaN/Inf/OOM | record point/resources; R3 decides whether the contract must change |
| infrastructure interruption | exact unit-boundary resume if frozen; otherwise new run id |
| structural/equivalence/scientific/safety gate fail | stop only the continuation named by the frozen gate |

A second same-class command failure or a repeated same-root engineering failure
produces one blocker. Never generate a new authorization, route-specific
validator, child task, or changed threshold as recovery.

## Locked Test And Archive

Locked test requires a previous closeout that explicitly authorizes its single
sealed command and identifies fixed architecture, weights, preprocessing,
operator, thresholds, fallback, output, and post-result no-tuning rule.

Commit intermediate compact evidence to the route branch. At terminal state or
an explicit major handoff, update the route card, evidence README/closeout,
index and changed family summary, then follow
`BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.
