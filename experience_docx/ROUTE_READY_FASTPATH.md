# Route-Ready Fast Path

Date: 2026-07-26

Status: generic runtime adopted after the CPU-only r2 E2E closeout in
`experiment_cards/2026-07-17-route-ready-fastpath-validation.md`. The
schema-v4 v4.3 engineering-review gate is adopted after fresh-process
acceptance; the six-tool surface and route schema are unchanged. Strict
post-integration validation passed on main with no bootstrap runtime files.
The route-authoring slimming update was adopted after 83/83 CPU-only cloud
tests at `2980c7970604c22a85242ca3ec669b030b08690b`; MCP, schema, and the
generic runtime bundle remain unchanged.

## Default Bundle

For a new route, author only:

1. one research-program contract under research_programs/;
2. one schema-v2 experiment spec under experiment_specs/; and
3. one Python entrypoint implementing contract(context_path) and
   run(context_path).

Run `experiment_spec_compiler.py --lint-all` once while authoring. It returns
all independently detectable errors as stable path/code/message JSON and never
writes derived files. After a clean lint, run `--write` once to deterministically emit the
schema-v6 route manifest, canonical scientific JSON, short rationale note,
runtime schema-v2 spec and any declared asset/capability/precision contracts.
The compiler does not select any scientific field. Do not hand-edit generated
files; route-ready and MCP recompile the committed sources and reject byte
drift. New typed asset manifests use schema 2. Historical experiment-spec,
scientific and asset schema 1 plus manifest schema 4/5 routes remain supported
and unmodified.

Do not create an evidence README. The canonical JSON is the immutable machine
scientific contract; the route note owns rationale only. Terminal interpretation belongs only in the science-fastpath
conclusion JSON, while the typed closeout and formal result files own machine
evidence.

Every operation uses the unchanged
`experience_docx/tools/run_route_operation.sh` from GitHub main. Do not create a
route-specific shell lifecycle, validator, dispatcher, watcher, authorization
file, receipt, closeout writer, or output-path wrapper.

Materialize only the operation currently authorized to run. Freeze later-stage
stop rules in the experiment spec, but do not prebuild unauthorized workload
code. For schema 6, add an authorized later operation by editing the one spec
and recompiling; prepare_next_operation.py remains a historical schema-4/5
helper and cannot become a second schema-6 writer.

## One Gate Before One Push

Stage the complete route bundle, then run exactly one static gate against the
commit that the index would create:

```text
git add <complete-route-bundle>
python3 experience_docx/tools/validate_route_ready.py \
  --repo . --operation <OPERATION_ID> --report /tmp/route-ready.json
```

The gate rejects unstaged or untracked input. It invokes the exact MCP
`parse_manifest`, validates every operation and canonical contract, checks the
short route note, compares the generic runtime bundle byte-for-byte
with current GitHub main, compiles Python/Bash syntax, validates runtime/asset
schemas, checks the entrypoint interface, and proves that all published
evidence and closeout filenames are route-wide unique.
For schema 6 it additionally validates the program/spec SHA values, route-family
mechanism authorization and exact deterministic regeneration of every derived
file.

Do not run separate `py_compile`, JSON formatting/parse, `git diff --check`,
card, manifest, runtime-spec, or asset-schema checks before this gate. They are
already included, and common independent card/monitor/engineering-terminal
authoring errors are reported together in the same invocation.

After `ROUTE_READY_OK`, make one commit and push once. For a route authored and
pushed in the current clean worktree, the default launch path is one
`convir_route_plan`, exact verification of the requested branch/commit and the
returned operation/output identity with zero mismatches, then one immediate
`convir_route_start` when start authorization is already `YES`. Do not insert
`convir_git_status` between push and plan by default; reserve it for an
uncertain checkout, a reopened task, or an evidence-sync worktree. Runtime
identity, cloud paths, resources, assets, device-aware no-data contract execution,
telemetry, timeout, evidence publication, and typed closeout are owned by MCP
plus the generic lifecycle. Do not repeat those checks manually. Observe at
the frozen expected end with `convir_route_finish`; an active healthy run needs
no resident model watcher. `RUNNING_VERIFIED` is the only nonterminal start
state that permits telling the user the experiment is normally running.
`LAUNCHED_PENDING_VERIFICATION` means only that the process exists; call one
later finish after the startup interval and report any early failure
immediately.
If start returns `START_STATE_UNKNOWN`, never create another plan or launch.
Repeat that same sealed start once: its built-in metadata-only recovery either
returns the original launch receipt, proves a clean unchanged retry, or stops
as ambiguous.

Route-ready and plan report capability-registry lookup. One unique exact match
of source commit, production-code SHA-256, checkpoint SHA-256, runtime-
environment SHA-256, device class and mechanically derived input-contract
SHA-256 makes contract execution unnecessary. The contract entrypoint remains
statically valid as the fail-closed fallback for a later miss. Device class is
verified on cloud before reuse, and reuse always reports scientific
authorization `NONE`.

From Windows, use fixed WSL argv such as `wsl.exe --exec /usr/bin/git -C
<absolute-wsl-path> ...`; do not use nested `bash -lc` quoting. Use the compact
authoritative snapshot first and read only its referenced files.

`--bootstrap-runtime-bundle` is valid only for the first infrastructure branch
that introduces this bundle. It still requires every already-present main file
to match exactly. Ordinary scientific routes must never use it.

## Entrypoint Contract

The runtime spec is the only operation-to-lifecycle adapter. It freezes the
entrypoint, timeout, expected wall time, evidence role, protected-data
permissions, recovery policy, environment, assets, and compact evidence files.
Paths are delivered through `RouteContext`; entrypoints never infer positional
paths or derive cloud workspaces.

Every external asset declares an `access_role` and whether it is available to
the CPU contract. Confirmation, canary, and sealed-final assets can never be
exposed to that phase, and their run-phase delivery requires the matching
runtime permission.

`contract(context_path)` is protected-data-free and runs before expensive work
unless the exact capability lookup above succeeds.
It uses `metadata_only`, `cpu_exact`, `cpu_reference_equivalent`, or
`gpu_synthetic_no_data` from an identity-bound capability profile, and must validate
the entrypoint's synthetic output/finalizer contract, and cannot create
`workload/` or touch confirmation, canary, or locked-test data. For scientific
schema 2, `run(context_path)` calls `write_gate_result` with exact typed gate
outcomes and cannot choose a state, decision or authorization; the generic
lifecycle applies the complete frozen decision table. Historical scientific
schema 1 continues to use `write_run_result`. A nonempty schema-2 workload must
also load and record the generic completed-unit ledger; finalization requires
exact `total_units` coverage. The generic lifecycle alone owns:

For algorithms whose termination or cost depends on formal problem size, the
runtime engineering contract freezes `cost_contract`. `same_scale_probe`
executes the formal iteration count and verifies wall-time and peak-memory
bounds. `fixed_linear_extrapolation` is accepted only when enums prove an exact
fixed-count production map, no candidate search, fixed termination, fixed or
bounded shapes and constant memory; its probe count, per-iteration ceiling,
fixed overhead, safety factor and formal bound are mechanically checked. Any
adaptive search, graph/matrix-size dependence or data-dependent termination
must use `same_scale_probe`. Free-form rationale cannot waive this gate. Small
functional fixtures remain useful for determinism but do not validate ETA or
launch readiness.

Long contract phases call `write_contract_progress` only with stage and bounded
iteration counts. The status contract forbids data ids, metrics, outcomes and
scientific values. On contract failure, the typed engineering diagnostic may
return only bounded failed-check names in addition to existing redacted control
metadata.

- fresh-output and identity preflight;
- exact asset verification and resolved asset delivery;
- timeout and process-group cleanup;
- fail-open metadata telemetry;
- write-once compact evidence publication; and
- the success closeout or generic
  `FAILED_ENGINEERING / null / NONE` closeout.

### Representative Engineering Fixture

A route that changes or exercises a real model, optimizer, or nontrivial
algorithm must use the smallest protected-data-free fixture that traverses the
exact production construction and changed execution path. Preserve the real
module graph, tensor rank/channel relationships, load/init/freeze and trainable
scope, forward/backward path, new-module gradient, and result finalizer that
the workload will use. Reduce batch and spatial size only where those
relationships remain valid; do not replace the changed path with a surrogate
mini-implementation.

Use the assertions in `route_engineering_fixture.py` for trainable scope,
no-op, finite tensor, nonzero-gradient, and microfit checks when they apply.
They standardize checks but never replace the exact production construction or
the route's frozen thresholds.

The fixture is an engineering check, not a scientific sample. Metadata-only,
schema-only, and ledger operations are exempt when their card records that no
model or numerical workload path exists. Run a representative fixture once per
unchanged exercised-code/entrypoint/runtime-spec/asset identity. A later stage
or repair may reuse that result when those identities and the exercised path
are unchanged; do not repeat it merely for reassurance. The
same-asymptotic-scale probe above remains additionally required when formal
cost or termination depends on problem size.
An unchanged fixture may be reused only through a six-field exact capability-
registry match. This is engineering qualification only; it cannot authorize
scientific PASS, protected data access or promotion.

The fixed output layout is:

```text
OUTPUT_PATH/
  control/
  contract/
  workload/
  heartbeat.json
  status.txt
  runtime.log
```

Closeout and compact evidence destinations are write-once. A failure before
output ownership is established may publish an engineering closeout, but it
must not alter an existing output directory.

## Recovery And Revalidation

The v1 fast path supports `none` and `complete_units`. Both launch into a new
output. `complete_units` requires one hash-bound unrestricted run-only file
asset named `completed_unit_ledger`. Every ledger row binds one unique unit and
input SHA-256 to an output asset/path/SHA-256; the generic API verifies imported
outputs, hashes each newly completed output itself, appends under a file lock and
fsyncs the ledger. Finalization requires exactly `total_units` verified rows.
It never means in-place reuse of an ambiguous output or trust in a count alone.
`exact_resume` is rejected at the staged gate until the schema-v4 control plane
has a receipt-bound transition that can prove it safely.

After an engineering failure, start/finish enters
`ENGINEERING_AUTO_REPAIR_AUTHORIZED` and evidence access is locked. Inspect
once, prepare one candidate with a new output identity, then run:

```text
python3 experience_docx/tools/validate_engineering_repair.py \
  --repo . --base <failed-route-commit> --operation <OPERATION_ID> \
  --snapshot worktree-candidate \
  --candidate-path <EXACT_CHANGED_PATH> [--candidate-path <EXACT_CHANGED_PATH> ...]
```

`worktree-candidate` copies only the explicitly named candidate paths into a
tool-owned temporary index, rejects any unlisted worktree change, creates the
ephemeral classification commit, and verifies that the real index stayed clean.
This is classifier input, not staging; real staging remains forbidden until the
gate returns `AUTO_REPAIR_ELIGIBLE`.

`AUTO_REPAIR_ELIGIBLE` proceeds without another repair prompt through the
normal route-ready/commit/push/plan/start path.
`SENSITIVE_REPAIR_REVIEW_REQUIRED` pauses before state-changing actions.
`archive` permits compact failure evidence sync but no repair/relaunch. The
explicit discard resolution is narrower: it requires a receipt-bound validated
engineering terminal, verified no scientific/protected data touch, exact paths,
an inactive session and post-delete checks. It never applies to scientific
terminals or shared assets.
cloud failure closeout remains required for provenance and diagnosis, but its
creation is not Git evidence sync. Do not change data, metrics, thresholds,
gates, or scientific scope. Re-run the staged gate only when the card,
manifest, runtime spec, entrypoint, asset manifest, or canonical runtime bundle
changed. A cloud contract pass is not repeated for unchanged code, and it
never constitutes a scientific result.
