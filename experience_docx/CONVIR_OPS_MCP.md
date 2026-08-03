# Convir Operations MCP

Date: 2026-08-02

Status: main-first snapshot server `5.9.0` retains exactly six tools and stable control
protocol schema 4. It reads immutable historical manifest schema 4/5 for status
and evidence provenance, while plan accepts only compiled manifest schema 6 and
runtime schema 2. New experiment specs and scientific contracts
use schema 3; typed asset manifests and other supporting contracts remain schema
2. Historical experiment/scientific schema 1/2 remains readable, is not
migrated and cannot pass route-ready, plan or start.

Version 5.9.0 adds a machine-enforced research-update binding for future routes:
the exact GitHub-main snapshot; either 1-8 `post_terminal` records or a zero-
terminal `program_foundation` bootstrap for a genuinely new program; explicit
bottleneck, stable literature identifiers and applicability limits, competing
falsifiable hypotheses, and the decision-value/time/cost basis for the chosen
design. The scientific contract changed, so the
compatibility id rotates when this contract changes. Prior rule-source commits
remain evidence provenance only and do not authorize historical reruns.

`convir-ops` is a restricted local stdio bridge to `convir-4090`. Its lifecycle
tools accept only a GitHub route branch, exact commit, and operation id; its
status tool accepts only bounded project/route identity inputs. It never
accepts an arbitrary command, remote path, metric, threshold, or scientific
verdict.

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
| `convir_git_status` | token-bounded GitHub-main project/route authority snapshot plus a separate local write binding; never a result reader |

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

For source experiment schema 3, the author does not provide
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
GitHub-main rules used for design. An exact bundle match passes directly. A
changed bundle passes only when current main's `RULE_COMPATIBILITY.json` keeps
the same compatibility id and explicitly names the prior `rules_commit`.
Incompatible scientific or authorization changes rotate that id and fail
closed. The compatibility file records a reviewed decision; it is not policy
authority and cannot silently relax a route contract.

For schema 6, the route author edits the research-program contract, one
experiment spec and the route entrypoint. The deterministic compiler emits the
remaining route files. Route-ready and planning recompile the committed sources
and reject any generated-file drift. Program governance distinguishes adjacent,
orthogonal and evidence-backed reopen mechanisms; it does not impose one global
experiment-count limit.

Scientific schema 3 binds the route to its triggering terminal evidence and
research update, then freezes finite typed outcomes for every gate and one
complete mutually exclusive decision table. The update must classify the
bottleneck, state 2-8 discriminating falsifiable hypotheses, and justify a
single-factor, multi-arm, factorial, multi-fidelity or group-sequential design
by decision value, expected time-to-decision, shared setup and worst-case
stopping cost. Authoring requires the binding snapshot to equal live main;
route-ready and planning later require it to remain an ancestor of current main
and verify terminal records from that frozen commit. Every operation shares the
same update. `program_foundation` additionally proves an archive-absent program
id, zero used attempts across its families and a first adjacent claim.
`validity_veto` makes failed
identity, integrity or coverage invalidate all scientific outcomes;
`inconclusive_only` precision cannot hide a decisive FAIL; descriptive gates
cannot affect a terminal. Route code writes `gate_outcomes`; the generic lifecycle derives state, decision,
authorization, next action and family effect. Each terminal action must be
operationally distinct. Historical scientific schema 2 retains these gate and
decision-table semantics; schema 1 retains its typed terminal writer.

The authoring compiler's atomic `--finalize` returns stable path/code/message
errors across independently checkable operations and source-text hygiene, then
writes the complete derived bundle and canonical nine-file runtime closure only
after proving the remote-tracking main equals live GitHub main. It mechanically
derives the capability input-contract SHA-256 while rejecting an explicit
mismatch. New
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
the old snapshot time and explicitly makes no current-health claim. Valid
workload `0/N` telemetry retains `N`, and workload stage takes precedence
over an earlier contract stage once workload telemetry has appeared. A closeout
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
`ENGINEERING_AUTO_REPAIR_AUTHORIZED` in a revisioned receipt whose complete
mutable state is HMAC-protected and returns
the failure immediately. Evidence list/fetch remain locked. The external
`validate_engineering_repair.py` gate distinguishes same-contract mechanical
repairs from sensitive scientific/data/model changes before commit/push/start.
Explicit `archive` still unlocks compact failure evidence but no relaunch.

If the failure occurred after workload completion and its typed phase is only
`evidence` or `finalize`, finish also accepts one finalization-only resolution.
It validates the candidate before atomically consuming the receipt's single
execution slot. The same commit may retry pure publication; a new commit must
pass the schema-6 terminal-adapter classifier with unchanged scientific,
program, operation, output and runtime identities. Cloud requires an inactive
session, revalidates the full unit ledger and every stable output before and
after, invokes only `finalize_existing`, and allows only declared review-facts
bytes to change. It never reruns workload. Failure or unknown state after slot
reservation cannot be retried under that receipt. An adapter-changed commit
cannot register a new capability identity from the source contract result.

Engineering diagnostics are bounded, redacted and control-only. Explicit
discard remains inside convir_route_finish and requires a validated receipt-
bound engineering closeout, verified no scientific/protected data touch, an
inactive session, exact derived repo/run/output/closeout identities and post-
delete absence. Scientific terminals and shared assets are never eligible.

Evidence tools allow only non-symlink top-level `.json/.csv/.md/.txt` files up
to 1 MiB. The evidence root must remain the exact receipt workspace, and every
access rechecks the validated closeout filename and SHA-256 stored in the
receipt before listing or fetching any file.
They require a scientific validated closeout or an explicit engineering
`archive` resolution and never stage, commit, or push. A cancellation receipt
does not unlock evidence tools.
Fetch defaults to stateless HMAC-bound inline UTF-8 pages. The continuation
binds receipt, ordered allowlist, file identities and byte offset; replay is
idempotent and creates no cursor file. `materialize` is explicit, requires a
bound local worktree, writes only canonical unstaged evidence destinations and
is used by archive workflows.
For a schema-6 scientific terminal, both evidence responses also return a compact
receipt-bound `archive_contract` naming the canonical closeout and conclusion
paths, schema-3 conclusion version and required fields. At least one selected
primary review fact must have a source-bound numeric point. This is a contract hint,
not a generated conclusion: the operator must author the JSON, and archive
derives the same canonical path when `--conclusion` is omitted. Engineering
archive responses do not expose a scientific conclusion contract.

The validated schema-6 terminal response from start/finish returns that same
`archive_contract` immediately, sets `archive_ready=false`, and supplies the
ordered action sequence `convir_evidence_list`, `convir_evidence_fetch`,
`author_scientific_conclusion`, `prepare_terminal_archive`. This removes an
ambiguous review-or-archive branch without duplicating evidence contents or
adding a seventh tool. Historical schema-4/5 and engineering terminals retain
their existing response contract.

`convir_git_status` supports `scope=project|route` without adding a seventh
tool. Project scope is the first call for a new task. It may omit `local_repo`
and then uses the registered dedicated main worktree; it proves live GitHub-main
freshness and returns compact policy, rule-compatibility and terminal-index
identities without inspecting a route worktree. It never infers a route or
scientific authorization.

Route scope preserves the legacy `local_repo, route_id, detail` call. It reads
the fresh main terminal index before local worktree state. A unique terminal
chain confirms an archived route independently of a dirty, missing or wrong
local manifest. A route with no terminal is `NO_TERMINAL_RECORD` only after a
GitHub route-branch manifest or local pre-push HEAD manifest confirms the id;
an arbitrary id remains `ROUTE_ID_UNRESOLVED`. The response separates
`authoritative_read_binding` from `local_write_binding`. Local dirtiness,
branch failure or manifest mismatch blocks only the latter, while stale main or
terminal-index identity conflict blocks the former. Require local write binding
before editing, commit, plan/start, repair or archive. Neither binding proves a
metric, verdict, completed workload or next-stage authorization.

The additive `SNAPSHOT_IDENTITY` phase receipt records scope, fresh main,
authoritative-read status, local-write status and one next action. It always
records `scientific_authorization=NOT_DERIVED`; authoritative closeout and
conclusion files must still be read for any authorization decision. It is a
control receipt, not evidence.

## Capability Reuse And Recovery

A schema-2 capability profile binds source commit, production-code SHA-256,
checkpoint SHA-256, runtime-environment SHA-256, device class and a mechanically
derived input-contract SHA-256 to committed asset identities. Route-ready,
plan and lifecycle query `capability_registry.jsonl` at the exact authoritative
main commit and validate only records matching the requested identity. Unrelated
historical corruption remains a registry-maintenance failure, not a route miss.
One unique exact match skips contract execution; any mismatch executes the frozen contract once. Cloud verifies the
actual CUDA compute-capability class before reuse. New qualification evidence is
compact and registered only by terminal archive. Every reuse result carries
`scientific_authorization: NONE`.

Every nonempty scientific schema-3 workload must finish with exact
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
tracking GitHub main. Never register a historical route or feature worktree.
After an update, fast-forward that dedicated worktree to verified GitHub main,
restart the host and verify version `5.9.0`, source SHA-256, exactly six tools,
schema 4/5/6 parsing, and the
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
