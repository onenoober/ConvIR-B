# Route-Ready Fast Path

Date: 2026-07-17

Status: generic runtime adopted after the CPU-only r2 E2E closeout in
`experiment_cards/2026-07-17-route-ready-fastpath-validation.md`. The
schema-v4 v4.3 engineering-review gate is adopted after fresh-process
acceptance; the six-tool surface and route schema are unchanged. Strict
post-integration validation passed on main with no bootstrap runtime files.

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

After `ROUTE_READY_OK`, make one commit, push once, then use one
`convir_route_plan` and one `convir_route_start`. Runtime identity, cloud paths,
resources, assets, CPU-only contract execution, telemetry, timeout, evidence
publication, and typed closeout are owned by MCP plus the generic lifecycle.
Do not repeat those checks manually. Observe at the frozen expected end with
`convir_route_finish`; an active healthy run needs no resident model watcher.
If start returns `START_STATE_UNKNOWN`, never create another plan or launch.
Repeat that same sealed start once: its built-in metadata-only recovery either
returns the original launch receipt, proves a clean unchanged retry, or stops
as ambiguous.

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
with a new output identity but does not authorize launch; `archive` permits
compact failure evidence sync but no repair/relaunch. Do not change data,
metrics, thresholds, gates, or scientific scope. Re-run the staged gate only
when the card, manifest, runtime spec, entrypoint, asset manifest, or canonical
runtime bundle changed. A cloud contract pass is not repeated for unchanged
code, and it never constitutes a scientific result.
