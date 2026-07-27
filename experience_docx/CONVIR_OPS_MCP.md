# Convir Operations MCP

Date: 2026-07-27

Status: governance-fastpath server `5.4.0` retains exactly six tools and stable control
protocol schema 4. It reads immutable historical manifest schema 4 and uses
immutable historical manifest schema 5 plus compiled manifest schema 6 and
runtime schema 2 for new routes. New experiment specs, scientific contracts and
typed asset manifests use schema 2; historical schema 1 remains readable and is
not migrated.

`convir-ops` is a restricted local stdio bridge to `convir-4090`. It accepts
only a GitHub route branch, exact commit, and operation id. It never accepts an
arbitrary command, remote path, metric, threshold, or scientific verdict.

For new routes, first require one staged `validate_route_ready.py` pass under
`ROUTE_READY_FASTPATH.md`, then commit/push once and call plan/start. The staged
gate shares this server's exact `parse_manifest`; MCP must not gain a second
route validator or a route-specific shell surface.

## Six Tools

| Tool | Purpose |
| --- | --- |
| `convir_route_plan` | validate and seal one committed operation; no cloud call |
| `convir_route_start` | run one sealed plan and return verified, pending, or early-failure state with a receipt |
| `convir_route_finish` | sealed finish, result-blind progress/terminal probe, receipt-bound cancellation, closeout validation and one same-contract repair state machine |
| `convir_evidence_list` | list eligible compact evidence for a receipt |
| `convir_evidence_fetch` | fetch an explicit allowlist with SHA-256 checks |
| `convir_git_status` | token-bounded worktree/GitHub audit plus authoritative route snapshot |

Do not add generic shell, SSH, cleanup, retry, watcher, commit, push,
authorization-file, validator or model-routing tools.
Internally generated, schema-bound bodies cross the fixed host boundary only as
one stdin script to fixed `/usr/bin/ssh` and `/bin/bash`; stdout and stderr are
drained with 64 KiB caps and every call has a hard timeout. No tool accepts the
body or any remote command from a caller. The separate `convirctl remote-script`
contract remains limited to unchanged Git-tracked scripts for manual actions.

## Manifest

Fixed path: `experience_docx/route_operations.json`; maximum 16 KiB. New-route top level:

```text
schema_version=6, route_id, rules_commit, route_card_relpath,
scientific_contract_relpaths, program_contract_relpath,
program_contract_sha256, experiment_spec_relpath, experiment_spec_sha256,
operations
```

One operation contains exactly:

```text
runner_relpath, mode, require_gpu, output_id, closeout_filename,
prior_closeout_relpath, prior_terminal_tuple, allowed_terminal_tuples,
workspace_policy, output_policy, monitor_profile, heartbeat_timeout_seconds,
min_free_gpu_mib, max_gpu_utilization_pct
```

For source experiment schema 2, the author does not provide
`allowed_terminal_tuples`. The compiler derives exactly the three scientific
tuples from the canonical terminal actions and adds only the generic
`FAILED_ENGINEERING / null / NONE` tuple.

`scientific_contract_relpaths` maps every operation id to one immutable canonical
JSON. Each canonical scientific JSON owns the question, population/grouping,
intervention, primary estimand, controls, uncertainty, gates, competing
explanation, terminal mapping and disabled actions. The <=8 KiB Markdown card
is only a rationale/pointer and is never parsed for scientific fields.
The exact route commit fixes the contract, card and runner. MCP derives their
blob/SHA values and the canonical-rule bundle digest directly from Git; route
authors do not copy these digests into the manifest. `rules_commit` records the
GitHub-main rules used for design. Planning accepts it only when its canonical
bundle still equals current main.

For schema 6, the route author edits the research-program contract, one
experiment spec and the route entrypoint. The deterministic compiler emits the
remaining route files. Route-ready and planning recompile the committed sources
and reject any generated-file drift. Program governance distinguishes adjacent,
orthogonal and evidence-backed reopen mechanisms; it does not impose one global
experiment-count limit.

Scientific schema 2 freezes finite typed outcomes for every gate and one
complete mutually exclusive decision table. `validity_veto` makes failed
identity, integrity or coverage invalidate all scientific outcomes;
`inconclusive_only` precision cannot hide a decisive FAIL; descriptive gates
cannot affect a terminal. Route code writes `gate_outcomes`; the generic lifecycle derives state, decision,
authorization, next action and family effect. Each terminal action must be
operationally distinct. Historical scientific schema 1 retains its typed
terminal writer.

The authoring compiler provides a write-free aggregate lint that returns stable
path/code/message errors across independently checkable operations. New
non-metadata authoring must also declare one machine-validated cost strategy.
Adaptive/nonlinear work uses `same_scale_probe`; the narrower
`fixed_linear_extrapolation` requires fixed-count exact production execution,
no search schedule, bounded shapes, constant memory and a conservative formal
bound. Historical runtime schema 1 and earlier schema-2 contracts remain
readable and are not migrated.

The first operation has no prior closeout and must be named by the canonical
contract. Every
later operation binds one prior closeout and exact
`state/decision/authorizes` tuple. No initial/intermediate authorization file is
valid.

## Finite State

Start checks resources before creating a fresh workspace and again immediately
before launch. Fresh workspaces use the immutable cloud anchor as a local shared
object seed and fetch only the exact route branch tip, avoiding a full GitHub
history clone on every route. Resource wait may reuse the unchanged plan. Any timeout after
the launch boundary becomes `START_STATE_UNKNOWN` and forbids a second launch.
One repeat of the same `convir_route_start` performs a bounded metadata-only
inspection. Exact repo/runner plus an active session, bound output, or valid
closeout recovers a receipt without launching again. Proven no-launch resets
the unchanged plan; an exact clean abandoned fresh workspace may be removed
only before any session/output/closeout exists. Ambiguous state stops after
that single inspection. A receipt is the only input for finish/evidence tools. Finish prefers
`heartbeat.json`, falls back to `status.txt`, and then to receipt launch time. A
stale active heartbeat is a bounded infrastructure warning: it never controls
the workload and leaves the receipt open for later closeout validation. A dead
session without closeout or a validated closeout closes `finish`. Healthy
receipts have a hard maximum of 64 observation windows, preventing an unbounded
watcher loop.

Pending and running sealed-finish results return `retry_after_seconds`,
`not_before_unix` and `expected_phase_end_unix`. Before not-before, another
sealed call returns its cached typed state without SSH or an observation-window
charge. ETA and not-before do not block operator control.

`observation_mode=progress_only` performs one immediate result-blind snapshot
with a separate maximum of 256 cloud observations and a 15-second minimum
interval. It returns only token-safe stage, completed/total units, exact-session
activity, heartbeat age/source, `snapshot_at_unix`, `cached`, and whether current
health is claimed. It never returns a metric, loss, sample/data id, action
distribution, gate outcome or scientific decision. A cached response preserves
the old snapshot time and explicitly makes no current-health claim. A closeout
detected before ETA returns `TERMINAL_DETECTED`, clears the sealed not-before
cache and allows immediate normal finish; it does not read or reveal the
scientific tuple. A dead exact session without closeout returns
`CLOSEOUT_MISSING` immediately. No resident watcher is allowed.

`operator_action=cancel` is also available before ETA and accepts no PID,
session, path or command argument from the caller. The receipt derives route,
run, commit, runner, repo, output, closeout and session. Cloud then validates
the lifecycle identity file, exact Git/runner identity, tmux pane command,
environment, owner and Linux start ticks, writes one idempotent cancellation
request, and signals only that lifecycle. The lifecycle gracefully terminates
its bound child process group and writes a cancellation closeout. After a
30-second grace period the controller may terminate only the previously
captured and revalidated process tree, waits another bounded interval, and
uses the lifecycle control schema to finalize the closeout if an older runner
cannot do so. PID reuse or identity drift fails closed. One same-request repeat
is allowed after unknown transport state.

The cancellation tuple is always
`CANCELLED_BY_OPERATOR / null / NONE`. It is a control terminal outside the
scientific contract's allowed tuples and never authorizes scientific
interpretation, engineering repair, archive, promotion, relaunch or reuse of
partial evidence. If a scientific or engineering closeout won the race before
the cancellation signal, that original terminal is preserved and returned.
Contract progress accepts only stage plus bounded completed/total iteration
counts. Contract failure diagnostics may add bounded failed-check tokens; data
identifiers, metrics, outcomes and scientific values are forbidden.

A successful launch command is not a successful experiment start. Start spends
one bounded monitor window and returns `RUNNING_VERIFIED` only after positive
machine-readable workload progress. It returns
`LAUNCHED_PENDING_VERIFICATION` for a live zero-progress process, or the typed
engineering closeout directly when startup fails.

A validated `FAILED_ENGINEERING / null / NONE` closeout does not enter the
scientific/archive path. Start/finish records
`ENGINEERING_AUTO_REPAIR_AUTHORIZED` in the HMAC-protected receipt and returns
the failure immediately. Evidence list/fetch remain locked. The external
`validate_engineering_repair.py` gate distinguishes same-contract mechanical
repairs from sensitive scientific/data/model changes before commit/push/start.
Explicit `archive` still unlocks compact failure evidence but no relaunch.

Engineering diagnostics are bounded, redacted and control-only. Explicit
discard remains inside convir_route_finish and requires a validated receipt-
bound engineering closeout, verified no scientific/protected data touch, an
inactive session, exact derived repo/run/output/closeout identities and post-
delete absence. Scientific terminals and shared assets are never eligible.

Evidence tools allow only top-level `.json/.csv/.md/.txt` files up to 1 MiB.
They require a scientific validated closeout or an explicit engineering
`archive` resolution and never stage, commit, or push. A cancellation receipt
does not unlock evidence tools.

## Capability Reuse And Recovery

A schema-2 capability profile binds source commit, production-code SHA-256,
checkpoint SHA-256, runtime-environment SHA-256, device class and a mechanically
derived input-contract SHA-256 to committed asset identities. Route-ready and
plan query `capability_registry.jsonl`. One unique exact match skips contract
execution; any mismatch executes the frozen contract once. Cloud verifies the
actual CUDA compute-capability class before reuse. New qualification evidence is
compact and registered only by terminal archive. Every reuse result carries
`scientific_authorization: NONE`.

Every nonempty scientific schema-2 workload must finish with exact
`total_units` coverage in the generic completion ledger. `complete_units`
remains a fresh-output transition and additionally requires a hash-bound
unrestricted run-only `completed_unit_ledger` file asset. The generic API and
lifecycle verify unique unit/input/output bindings and each referenced output
SHA-256, fsync newly completed rows and require exact `total_units` coverage.
A count without that ledger and output evidence is not reusable.

The schema-2 terminal index stores SHA-256 bindings for the contract, closeout,
conclusion, formal results and compact launch-contract bundle, plus the direct
parent closeout and terminal tuple. `convir_git_status` verifies those blobs and
selects the unique leaf of a complete parent chain. Branches, cycles, missing
parents, disconnected terminals and hash mismatches fail closed. Legitimate
multi-operation chains remain readable.

## Registration

Register one `convir_ops` server pointing at one clean dedicated worktree
tracking GitHub main. After an update, restart the host and verify version
`5.4.0`, source SHA-256, exactly six tools, schema 4/5/6 parsing, and the
startup/progress/cancel/repair/discard states.

The stdio server is a long-lived process and never hot-updates from Git. A
running task may continue on its already loaded source. Normal host restart or a
new task loads the updated main worktree; bounded acceptance uses a fresh
process and must not terminate another active task merely to force activation.

Only an engineering receipt carrying `v43_migrated_at` may change its automatic
legacy `archive` resolution to an explicit user-selected `repair`. A normal
v4.3 archive decision remains terminal and cannot be reopened.

## Historical Adoption Records

The records below preserve provenance for prior server versions only. They do
not define the current server version, registration target, tool surface, or
operating workflow; those are specified above.

The 2026-07-16 adoption audit verified the registered executable at
`main@dca94d71c9fe73e4e93910b0587927c79ab7023c`: version `4.1.0`, source
SHA-256 `dc07ac60056b5e7da52f045419f196c82f6a19a64255408504056b1329ecc2ae`,
schema v4, and exactly six tools. Do not terminate an active task's stdio server
to force reload; use a new app/task session or an isolated-state validation of
the exact registered executable.

The pre-activation v4.2 candidate at
`980821176f09514d913f4ad0507e494b3c45971b` passed 73 cloud tests with schema
v4, exactly six tools, bounded unknown-start receipt recovery, clean-abandoned-
workspace retry, shared-seed checkout, and zero model calls. The code is on
main.

Final acceptance on 2026-07-17 verified the fresh registered process at
`/home/ubuntu/workspace/ConvIR-B-operations-v4/experience_docx/tools/convir_ops_mcp.py`.
Its source SHA-256
`f84330ffc1ffe5b6973f710078e81bfb35bd4ffccab97e15096397e6e75d6e8a`
matches GitHub main exactly, the live surface contains exactly six tools, and
three consecutive read-only status calls returned the same clean/fresh main
identity without Git mutation. A new cloud execution at
`8a93cbb9af9c9731e2fe118cfb241edaf22067cb` again passed all 73 tests with
zero model calls. This closes candidate status as `CONVIR_OPS_V4_2_ADOPTION`;
see `experiment_logs/convir_ops_v4_2_final_acceptance_20260717/`.
