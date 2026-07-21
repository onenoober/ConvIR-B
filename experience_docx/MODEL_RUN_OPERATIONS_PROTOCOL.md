# Model Run Operations Protocol

Date: 2026-07-16

## Launch Order

```text
one staged route-ready gate -> one exact route commit/push -> one MCP plan ->
dynamic cloud preflight -> generic runner/contract -> bounded observation ->
typed closeout -> one science-fastpath archive OR engineering review decision
```

Local WSL remains syntax/compile-only. Before launch, the staged gate and
MCP/generic lifecycle jointly verify exact route HEAD, workspace policy,
runner/assets, prior closeout, data role, locked-test policy, GPU floor, new
output/recovery identity, status/log/closeout paths and protected-data
permissions. Never substitute a commit, data, split, checkpoint, threshold,
output or stage silently. Do not manually repeat these machine-owned checks.

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
identity/asset/output preflight, executes the declared protected-data-free
engineering contract before expensive work, captures stdout/stderr, enforces timeout/protected-data/result
contracts, publishes compact evidence write-once, and writes one closeout bound
to route id, run id, route commit, runner SHA-256, evidence role, and allowed
terminal tuple.

The contract phase validates route-specific input/no-op/shape/finite and
output/finalizer behavior and cannot create `workload/`. Runtime schema 2
requires one identity-bound mode: metadata-only, CPU exact, CPU reference
equivalent, or GPU synthetic with no scientific data, training or result.
Scientific computation belongs only to the run phase. Route-specific shell
lifecycle, closeout, telemetry, output-path, timeout, and evidence-copy code is
not allowed by default.

The representative engineering fixture and its metadata-only exemptions are
defined once in `ROUTE_READY_FASTPATH.md`. Generic assertions are provided by
`route_engineering_fixture.py` and documented in `ROUTE_FLOW_TOOLS.md`. Reuse
that fixture result for an unchanged exercised path; do not add another
route-specific smoke layer.

For any operation whose cost or termination depends on sample count, group
count, candidate count, graph size, search depth, or matrix dimension, the
engineering contract must also execute a protected-data-free synthetic probe at the same
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

`convir_route_start` creates the cloud process and immediately spends one
bounded observation window. `RUNNING_VERIFIED` is the only nonterminal state
that may be reported as "the experiment is running"; it requires positive
machine-readable workload progress. `LAUNCHED_PENDING_VERIFICATION` means only
that a process exists and must not be described as successful entry into the
experiment. An engineering closeout during this window is returned directly by
`start`, so preflight or unit-zero failure cannot stay silent until the ETA.

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
`convir_route_start` or `convir_route_finish` validates that tuple it returns
`ENGINEERING_AUTO_REPAIR_AUTHORIZED`, not a scientific/archive authorization.
Perform at most one receipt-bound, read-only diagnosis and prepare one repair:

```text
FAILED_ENGINEERING
  -> ENGINEERING_AUTO_REPAIR_AUTHORIZED
  -> AUTO_REPAIR_ELIGIBLE -> one same-contract repair cycle
  -> SENSITIVE_REPAIR_REVIEW_REQUIRED -> user decision
  -> explicit archive -> ENGINEERING_ARCHIVE_AUTHORIZED
```

The MCP keeps evidence locked. `validate_engineering_repair.py` permits only a
new output identity, immutable-identity file/Git path relocation, import/symbol
binding repair with unchanged arguments/control flow, and protected-data-free
contract fixture additions. It rejects data-directory changes and any runtime
spec, permission, seed/budget, algorithm constant/control-flow, model or asset
identity change. An eligible repair uses the normal single route-ready gate,
commit/push, plan and existing start authorization without another user repair
prompt. A sensitive repair pauses before commit/push/start. `archive` unlocks
only compact failure evidence. A repeated same-root failure stops for review.

Failure closeouts must retain the identities of assets successfully verified
before the failure. An empty list means verification did not complete, not
merely that the workload failed later.

Raw logs, checkpoints, images, arrays, predictions and large tables remain in
the cloud run root. GitHub retains the complete compact text evidence needed to
reproduce the terminal decision, as defined by `SCIENCE_FASTPATH.md`. After a
validated typed closeout, do not call finish again or perform heartbeat/status,
branch, worktree or output cleanup as part of experiment completion.
