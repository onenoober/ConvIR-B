# Model Run Operations Protocol

Date: 2026-07-27

## Launch Order

```text
one staged route-ready gate -> one exact route commit/push -> one sealed MCP plan ->
dynamic cloud preflight -> generic runner/contract -> bounded observation ->
typed closeout -> one science-fastpath archive OR engineering review decision
```

Local WSL remains syntax/compile-only. Before launch, the staged gate and
MCP/generic lifecycle jointly verify exact route HEAD, workspace policy,
runner/assets, prior closeout, data role, locked-test policy, GPU floor, new
output/recovery identity, status/log/closeout paths and protected-data
permissions. Never substitute a commit, data, split, checkpoint, threshold,
output or stage silently. Do not manually repeat these machine-owned checks.
Compiler, route-ready, plan and launch each bind the exact live GitHub-main
commit. A stale remote-tracking ref fails closed, and launch fetches that exact
main into a dedicated runtime ref before the lifecycle reads canonical runtime
or capability-registry bytes.
If live main moves after plan, start requires one fresh plan; compatibility may
preserve prior contract provenance but never mutates or extends a signed plan.

`min_free_gpu_mib` is the evidence-bound minimum free memory required for the
exact capability and cost contract to attempt execution, not a comfortable or
preferred-capacity target. A GPU that meets this floor and the frozen
utilization limit is resource-eligible for the attempt. Any higher comfortable
or preferred free-memory range may guide scheduling preference only; it must
not create a second blocking threshold. The protected-data-free synthetic
contract and existing peak-memory and cost bounds still determine whether the
execution path is qualified to continue.
When several devices are eligible, scheduling chooses the device with most free
memory, then lowest utilization, then lowest index. The selected device is
rechecked immediately before the tmux launch boundary; this narrows but cannot
eliminate races from external, non-cooperating GPU processes.

For the default path, `validate_route_ready.py` performs the complete static
check once before commit. MCP plan/start and the generic contract phase own the
dynamic checks once after push. Do not insert another route-specific checklist,
shell preflight, validator, or path wrapper between them.

Before plan reads the route manifest or any scientific contract, its built-in
control self-check compares `loaded_control_commit`,
`configured_worktree_head`, `live_main_commit`, the loaded/configured/main
validator bundle SHA-256 values, the module-origin manifest and the engineering
timeout policy. A mismatch is `CONTROL_PLANE_STALE` or
`VALIDATOR_BUNDLE_MISMATCH`; an unavailable identity read is
`CONTROL_SELF_CHECK_FAILED`. These states create no plan token and consume no
plan attempt. Only `PLAN_SEALED` is counted. They are never evidence that the
experiment contract is invalid.

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

New scientific schema-3 entrypoints publish typed gate outcomes only. The
lifecycle loads the canonical contract, resolves its complete decision table,
and alone chooses the state, decision, authorization, next action and family
effect. Historical scientific schema-1 terminal writers remain supported.
Every nonempty schema-3 workload records unique completed-unit input/output
identities, and the lifecycle refuses terminal derivation until the ledger
covers exactly `total_units`.

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
that fixture result for an unchanged exercised path only through a unique exact
six-field capability-registry match; do not add another route-specific smoke
layer. Route-ready and plan expose the match, cloud verifies device class, and
the lifecycle skips the contract execution entirely. A mismatch runs the frozen
contract once and publishes a qualification for terminal-archive registration.
Reuse remains engineering-only and authorizes no science or protected data.

For any operation whose cost or termination depends on sample count, group
count, candidate count, graph size, search depth, or matrix dimension, the
engineering contract freezes a machine-readable cost strategy. Adaptive search,
data-dependent termination, dynamic candidates and graph/matrix-size work must
execute a protected-data-free `same_scale_probe` at the formal iteration count.
`fixed_linear_extrapolation` is narrower: it requires the exact production path,
a fixed iteration map, no candidate schedule, fixed termination, bounded batch/
shape, constant memory, a smaller frozen probe, conservative safety factor and
a mechanically sufficient formal wall-time bound. Both strategies report exact
observed iterations, wall time and peak memory. A tiny functional fixture proves
determinism/correctness only and cannot support an ETA or launch-ready claim.
Do not compensate for an unbounded algorithm by merely increasing timeout.

For long operations, follow `GENERIC_RUN_MONITORING_PROTOCOL.md`: a fail-open
metadata-only sidecar may atomically replace `heartbeat.json`, while the runner
appends milestone-only `status.txt`. Telemetry never reads scientific outputs
or controls the workload; the runner remains the sole closeout owner.

Recovery uses a new output. A `complete_units` route may consume only the
predeclared hash-bound `completed_unit_ledger` file asset. Its unique unit/input/
output-asset/output-path/output-SHA records are verified against actual files;
new rows are file-locked and fsync'd after the generic API hashes the completed
output. An incomplete or mismatched unit restarts. In-place `exact_resume` is
not part of fast-path v1. Recovery cannot reveal confirmation results early or
change the scientific contract.

## Observe And Stop

`convir_route_start` creates the cloud process and immediately spends one
bounded observation window. `RUNNING_VERIFIED` is the only nonterminal state
that may be reported as "the experiment is running"; it requires positive
machine-readable workload progress. `LAUNCHED_PENDING_VERIFICATION` means only
that a process exists and must not be described as successful entry into the
experiment. An engineering closeout during this window is returned directly by
`start`, so preflight or unit-zero failure cannot stay silent until the ETA.

`convir_route_finish` observes one window: `short` is 30 seconds and `standard`
is 60 seconds. Pending/verified results return retry-after, not-before and
expected-phase-end timestamps. A repeated call before not-before returns the
cached typed state without a cloud call or observation-budget charge. A healthy
active run does not require a watcher, but the receipt holder may request an
immediate `progress_only` snapshot at any time. This path has its own finite
budget and minimum interval, exposes only result-blind stage/count/activity/
heartbeat metadata, and labels cached state with its original snapshot time.
It also probes for a closeout or a dead session, so ETA is never an embargo on
terminal/failure discovery.
A stale active heartbeat is an infrastructure warning and leaves the receipt
open for later terminal closeout validation; it never stops or restarts the
workload. A dead session without closeout gets one engineering inspection and
then stops. Never create a watcher loop or task per poll.

The human operator may explicitly cancel an active run through the same receipt.
Cancellation is not a scientific gate and does not require waiting for ETA. It
must verify the exact route/run/commit/runner/workspace/output/session and Linux
process identity, write the request before signaling, attempt graceful lifecycle
termination first, and revalidate the captured process tree before bounded
escalation. PID-only cancellation is forbidden. The typed terminal is
`CANCELLED_BY_OPERATOR / null / NONE`; completed-unit count is retained for
audit, but partial outputs are not interpretable or reusable and evidence tools
remain locked. Cancellation is idempotent and never becomes
`FAILED_ENGINEERING`. If the route reached a genuine closeout first, preserve
that closeout rather than overwriting it.

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
binding repair with unchanged arguments/control flow, protected-data-free
contract fixture additions, and schema-6 compiler-deterministic regeneration
of synchronized entrypoint asset/capability identities. It rejects
data-directory changes and any runtime
spec, permission, seed/budget, algorithm constant/control-flow, model or asset
identity change. An eligible repair uses the normal single route-ready gate,
commit/push, plan and existing start authorization without another user repair
prompt. A sensitive repair pauses before commit/push/start. `archive` unlocks
only compact failure evidence. A repeated same-root failure stops for review.
Before classification, `worktree-candidate` uses only an isolated temporary Git
index and exact path allowlist; the real index must remain clean.

A post-workload failure whose typed phase is only `evidence` or `finalize` has
one narrower resolution: receipt-bound finalization repair. Candidate
classification must succeed before the single execution opportunity is
consumed. The original session must be inactive; the original result, complete
unit ledger and every ledger-bound output are verified before and after. A
changed route commit may alter only an explicit terminal-adapter function and
its compiler-synchronized identity. Runtime then invokes only
`finalize_existing`, permits byte changes only in declared review-facts output,
and writes provenance for source and finalization commits without rerunning the
workload. An adapter-changed commit cannot register a new capability identity
from the source contract result. Any workload/data/model/metric/gate/threshold change, incomplete
ledger, output drift, or repeated finalization failure stops for review.

Failure closeouts must retain the identities of assets successfully verified
before the failure. An empty list means verification did not complete, not
merely that the workload failed later. Contract failures also retain bounded
failed-check names when available; no scientific value, sample identifier or
raw log is eligible for that diagnostic.

Raw logs, checkpoints, images, arrays, predictions and large tables remain in
the cloud run root. GitHub retains the complete compact text evidence needed to
reproduce the terminal decision, as defined by `SCIENCE_FASTPATH.md`. After a
validated typed closeout, do not call finish again or perform heartbeat/status,
branch, worktree or output cleanup as part of experiment completion.
Receipt-bound evidence review defaults to stateless, HMAC-bound inline UTF-8
pages. Repeating the same continuation is idempotent and performs no worktree
write. Explicit materialization is reserved for archive workflows. New
schema-6 archive receipts require a schema-3 conclusion with at least one
source-bound numeric primary fact; historical schema-2 conclusions remain
readable but cannot satisfy that new receipt contract.
For future schema-3 scientific terminals, the generic lifecycle seals regular
files below `contract/` and `workload/` in
`control/raw_artifact_manifest.jsonl`; the closeout binds a compact GitHub
receipt. Symlinks, special files and identity changes fail closed.
