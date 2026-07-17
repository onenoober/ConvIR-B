# Route-Ready Fast Path

Date: 2026-07-17

Status: generic runtime adopted after the CPU-only r2 E2E closeout in
`experiment_cards/2026-07-17-route-ready-fastpath-validation.md`. The
schema-v4 v4.3 engineering-review gate is adopted after fresh-process
acceptance; the six-tool surface and route schema are unchanged. Strict
post-integration validation passed on main with no bootstrap runtime files.
The route-authoring slimming update was adopted after 83/83 CPU-only cloud
tests at `2980c7970604c22a85242ca3ec669b030b08690b`; MCP, schema, and the
generic runtime bundle remain unchanged.

## Default Bundle

For a new route, create only:

1. one launch-ready route card;
2. one schema-v4 `route_operations.json`;
3. one runtime spec per listed operation under `route_runtime_specs/`;
4. one Python entrypoint implementing `contract --context <json>` and
   `run --context <json>`;
5. one typed asset manifest under `route_assets/` only when external assets are
   required; and
6. one evidence-directory README.

Every operation uses the unchanged
`experience_docx/tools/run_route_operation.sh` from GitHub main. Do not create a
route-specific shell lifecycle, validator, dispatcher, watcher, authorization
file, receipt, closeout writer, or output-path wrapper.

Materialize only the operation currently authorized to run. Freeze the later
stage sequence and stop rules in the route card, but do not prebuild dormant
runtime specs, entrypoints, or asset manifests. Add a later operation only
after its prerequisite closeout authorizes it, unless it is already authorized
and shares the exact frozen implementation without result-dependent changes.

## One Gate Before One Push

Stage the complete route bundle, then run exactly one static gate against the
commit that the index would create:

```text
git add <complete-route-bundle>
python3 experience_docx/tools/validate_route_ready.py \
  --repo . --operation <OPERATION_ID> --report /tmp/route-ready.json
```

The gate rejects unstaged or untracked input. It invokes the exact MCP
`parse_manifest`, validates every operation in the manifest, runs the existing
launch-ready card validator, compares the generic runtime bundle byte-for-byte
with current GitHub main, compiles Python/Bash syntax, validates runtime/asset
schemas, checks the entrypoint interface, and proves that all published
evidence and closeout filenames are route-wide unique.

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
identity, cloud paths, resources, assets, CPU-only contract execution,
telemetry, timeout, evidence publication, and typed closeout are owned by MCP
plus the generic lifecycle. Do not repeat those checks manually. Observe at
the frozen expected end with `convir_route_finish`; an active healthy run needs
no resident model watcher.
If start returns `START_STATE_UNKNOWN`, never create another plan or launch.
Repeat that same sealed start once: its built-in metadata-only recovery either
returns the original launch receipt, proves a clean unchanged retry, or stops
as ambiguous.

From Windows, use fixed WSL argv such as `wsl.exe --exec /usr/bin/git -C
<absolute-wsl-path> ...`; do not use nested `bash -lc` quoting. Read the known
index, family summary, route card, and route evidence directory directly. Do
not run broad repository searches when those authoritative paths already
identify the required files.

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

`contract(context_path)` is CPU-only, runs before expensive work, must validate
the entrypoint's synthetic output/finalizer contract, and cannot create
`workload/` or touch confirmation, canary, or locked-test data. `run(context_path)`
owns only route science and writes one typed run result. The generic lifecycle
alone owns:

For algorithms whose termination or cost depends on formal problem size, the
contract must include a protected-data-free same-asymptotic-scale probe and
verify a frozen iteration/time/memory bound. Small functional fixtures remain
useful for determinism but do not validate ETA or launch readiness.

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

The fixture is an engineering check, not a scientific sample. Metadata-only,
schema-only, and ledger operations are exempt when their card records that no
model or numerical workload path exists. Run a representative fixture once per
unchanged exercised-code/entrypoint/runtime-spec/asset identity. A later stage
or repair may reuse that result when those identities and the exercised path
are unchanged; do not repeat it merely for reassurance. The
same-asymptotic-scale probe above remains additionally required when formal
cost or termination depends on problem size.

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
output. `complete_units` means a new run may consume a preregistered, verified
complete-unit asset; it never means in-place reuse of an ambiguous output.
`exact_resume` is rejected at the staged gate until the schema-v4 control plane
has a receipt-bound transition that can prove it safely.

After an engineering failure, finish enters `ENGINEERING_REVIEW_REQUIRED` and
evidence access is locked. Inspect once and pause for the user's explicit
`repair` or `archive` choice. `repair` permits one semantic-preserving repair
with a new output identity but does not authorize launch or failed-run Git
sync; after the repair passes, sync the successful replacement evidence.
`archive` permits compact failure evidence sync but no repair/relaunch. The
cloud failure closeout remains required for provenance and diagnosis, but its
creation is not Git evidence sync. Do not change data, metrics, thresholds,
gates, or scientific scope. Re-run the staged gate only when the card,
manifest, runtime spec, entrypoint, asset manifest, or canonical runtime bundle
changed. A cloud contract pass is not repeated for unchanged code, and it
never constitutes a scientific result.
