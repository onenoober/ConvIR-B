# Model Run Operations Protocol

Date: 2026-07-16

## Launch Order

```text
one staged route-ready gate -> one exact route commit/push -> one MCP plan ->
dynamic cloud preflight -> generic runner/contract -> bounded observation ->
typed closeout -> compact archive
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
cause. `START_STATE_UNKNOWN` requires one inspection and no blind retry.
Resource wait may retry the unchanged prelaunch plan.

Raw logs, checkpoints, images, arrays, predictions and large tables remain in
the cloud run root. Only curated compact text evidence is eligible for Git.
