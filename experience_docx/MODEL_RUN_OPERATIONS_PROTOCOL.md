# Model Run Operations Protocol

Date: 2026-07-16

## Launch Order

```text
one staged route-ready gate -> one exact route commit/push -> one MCP plan ->
dynamic cloud preflight -> generic runner/contract -> bounded observation ->
typed closeout -> scientific archive OR engineering review decision
```

Local WSL remains syntax/compile-only. Before launch, verify exact route HEAD,
clean/fresh or exact-continuation workspace, runner/assets, prior closeout, data
role, locked-test policy, GPU floor, new output/recovery asset, status, heartbeat,
log and closeout paths. Never substitute a commit, data, split, checkpoint,
threshold, output, or stage silently.

For the default path, `validate_route_ready.py` performs the complete static
check once before commit. MCP plan/start and the generic contract phase own the
dynamic checks once after push. Do not insert another route-specific checklist,
shell preflight, validator, or path wrapper between them.

## Generic Runner

Every route-ready operation uses the unchanged
`experience_docx/tools/run_route_operation.sh` and one declarative runtime spec.
The route-specific Python entrypoint implements only `contract --context` and
`run --context`; it receives every path and verified asset through
`RouteContext`. The generic lifecycle uses the explicit cloud Python, performs
identity/asset/output preflight, executes the CPU-only synthetic contract before
expensive work, captures stdout/stderr, enforces timeout/protected-data/result
contracts, publishes compact evidence write-once, and writes one closeout bound
to route id, run id, route commit, runner SHA-256, evidence role, and allowed
terminal tuple.

The contract phase validates route-specific input/no-op/shape/finite and
output/finalizer behavior that is safe on CPU, and cannot create `workload/`.
Scientific computation belongs only to the run phase. Route-specific shell
lifecycle, closeout, telemetry, output-path, timeout, and evidence-copy code is
not allowed by default.

The representative engineering fixture and its metadata-only exemptions are
defined once in `ROUTE_READY_FASTPATH.md`. Generic assertions are provided by
`route_engineering_fixture.py` and documented in `ROUTE_FLOW_TOOLS.md`. Reuse
that fixture result for an unchanged exercised path; do not add another
route-specific smoke layer.

For any operation whose cost or termination depends on sample count, group
count, candidate count, graph size, search depth, or matrix dimension, the CPU
contract must also execute a protected-data-free synthetic probe at the same
asymptotic scale as the formal workload. Freeze and check maximum iterations,
wall time and peak-memory class before launch. A tiny functional fixture proves
determinism/correctness only and cannot support an ETA or launch-ready claim.
Do not compensate for an unbounded algorithm by merely increasing timeout.

For long operations, follow `GENERIC_RUN_MONITORING_PROTOCOL.md`: a fail-open
metadata-only sidecar may atomically replace `heartbeat.json`, while the runner
appends milestone-only `status.txt`. Telemetry never reads scientific outputs
or controls the workload; the runner remains the sole closeout owner.

Recovery uses a new output. A `complete_units` route may consume only a
predeclared complete-unit asset with matching identity; an incomplete unit
restarts. In-place `exact_resume` is not part of fast-path v1. Recovery cannot
reveal confirmation results early or change the scientific contract.

## Observe And Stop

`convir_route_finish` observes one window: `short` is 30 seconds and `standard`
is 60 seconds. A healthy active run may be observed again near its frozen ETA.
A stale active heartbeat is an infrastructure warning and leaves the receipt
open for later terminal closeout validation; it never stops or restarts the
workload. A dead session without closeout gets one engineering inspection and
then stops. Never create a watcher loop or task per poll.

Failure classes stay distinct: command/transport, preflight/resource,
engineering/runtime, evidence/closeout, and scientific/safety gate. Use one
deterministic command correction and one engineering repair cycle per root
cause. `START_STATE_UNKNOWN` forbids a second launch. Repeat the same sealed
start call once only to run its metadata-only recovery inspection; it may
recover a receipt, prove a clean retry state, or stop as ambiguous.
Resource wait may retry the unchanged prelaunch plan.

## Engineering Failure State Machine

`FAILED_ENGINEERING` always has `decision:null` and `authorizes:NONE`. When
`convir_route_finish` validates that tuple it returns
`ENGINEERING_REVIEW_REQUIRED`, not a scientific/archive authorization. Perform
at most one receipt-bound, read-only root-cause inspection, report the exact
failure and then stop for an explicit user decision:

```text
FAILED_ENGINEERING
  -> ENGINEERING_REVIEW_REQUIRED
  -> repair  -> ENGINEERING_REPAIR_AUTHORIZED
  -> archive -> ENGINEERING_ARCHIVE_AUTHORIZED
```

Before that decision, do not call evidence list/fetch, create or edit a repair
bundle, stage/commit/push, update the route card/index/family summary, create a
new plan, or start a workload. The MCP enforces the evidence lock. `repair`
allows one deterministic root-cause repair with the scientific/data contract
unchanged; it does not authorize launch, archive, or in-place output reuse.
Freeze a new code commit and output identity, rerun the appropriate static and
representative-scale engineering gates, and request separate start
authorization. `archive` unlocks only compact failure evidence and keeps
relaunch blocked. A repeated same-root engineering failure stops and returns to
the user; it does not consume scientific evidence or become scientific `FAIL`.

Failure closeouts must retain the identities of assets successfully verified
before the failure. An empty list means verification did not complete, not
merely that the workload failed later.

Raw logs, checkpoints, images, arrays, predictions and large tables remain in
the cloud run root. Only curated compact text evidence is eligible for Git.
