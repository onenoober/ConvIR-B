# Route Flow Tools

Date: 2026-07-17

Status: adopted after 97/97 CPU-only cloud tests at
`b72663e546239289a6a1679ae0b404b46bf0e1a5`; MCP, schema, and generic runtime
files are unchanged.

These helpers remove mechanical authoring and archival work without changing
scientific gates or launch authorization. Request files are temporary operator
inputs and must stay outside the route worktree, normally under `/tmp`.

## Classify One Engineering Repair

After `ENGINEERING_AUTO_REPAIR_AUTHORIZED`, diagnose once, prepare one staged
candidate with a new output identity, and run:

```text
python3 experience_docx/tools/validate_engineering_repair.py \
  --repo . --base <failed-route-commit> --operation <OPERATION_ID>
```

The gate freezes route/runtime/data/permission/seed/budget contracts and asset
identities. It allows immutable-identity file/Git path relocation, output
identity replacement, import/symbol qualification repair, protected-data-free
contract fixtures, and a standard route-card repair note. It returns
`SENSITIVE_REPAIR_REVIEW_REQUIRED` for algorithm/control-flow changes,
directory/data path changes, model/checkpoint identity changes, runtime-spec
changes, unexpected files, or unparseable input. Only
`AUTO_REPAIR_ELIGIBLE` may proceed automatically to route-ready and launch.

## Advance One Authorized Operation

After a committed closeout authorizes exactly one next operation, prepare one
temporary schema-1 request containing:

```json
{
  "schema_version": 1,
  "operation_id": "A0",
  "operation": {
    "mode": "a0",
    "require_gpu": true,
    "output_id": "a0-r1",
    "closeout_filename": "a0_closeout.json",
    "allowed_terminal_tuples": [
      {"state": "COMPLETED_GATE_PASS", "decision": "A0_PASS", "authorizes": "A1_REVIEW"},
      {"state": "COMPLETED_GATE_FAIL", "decision": "A0_FAIL_STOP", "authorizes": "NONE"},
      {"state": "FAILED_ENGINEERING", "decision": null, "authorizes": "NONE"}
    ],
    "workspace_policy": "fresh_route",
    "output_policy": "new",
    "monitor_profile": "standard",
    "heartbeat_timeout_seconds": 300,
    "min_free_gpu_mib": 12000,
    "max_gpu_utilization_pct": 20
  },
  "runtime_spec": {
    "entrypoint_relpath": "experience_docx/tools/a0.py",
    "asset_manifest_relpath": "experience_docx/route_assets/A0.json",
    "timeout_seconds": 14400,
    "expected_wall_seconds": 7200,
    "total_units": 768,
    "evidence_role": "development_screening",
    "resume_policy": "none",
    "protected_data_permissions": {
      "allow_confirmation": false,
      "allow_canary": false,
      "allow_locked_test": false
    },
    "environment": {},
    "evidence_files": [
      {
        "source_relpath": "workload/a0_summary.json",
        "destination_filename": "a0_summary.json",
        "required": true,
        "max_bytes": 131072
      }
    ]
  }
}
```

Apply once:

```text
python3 experience_docx/tools/prepare_next_operation.py \
  --repo . --request /tmp/next-operation.json \
  --prior-closeout experience_docx/experiment_logs/<route-id>/<prior>_closeout.json \
  --apply
```

The helper reads the prior closeout and current manifest from committed `HEAD`,
requires `COMPLETED_GATE_PASS` to authorize the exact requested operation,
preserves route/rules/card identity, refuses reused operation/output/closeout
identities, and writes only `route_operations.json` plus the new runtime spec.
It does not edit the route card, write the entrypoint/assets, stage, commit,
plan, or start. Finish those route-specific files, then run the one staged
route-ready gate.

## Build A Typed Asset Manifest

For a runtime spec that declares an asset manifest, prepare one temporary
schema-1 request. Every asset path is the runtime path; `identity` is either a
predeclared identity or an explicitly permitted local identity source.

```json
{
  "schema_version": 1,
  "assets": [
    {
      "id": "metadata",
      "kind": "file",
      "path": "/runtime/metadata.json",
      "access_role": "unrestricted",
      "contract_access": true,
      "identity": {"local_file": "/local/metadata.json"}
    },
    {
      "id": "development_data",
      "kind": "file",
      "path": "/runtime/development.bin",
      "access_role": "development_screening",
      "contract_access": false,
      "identity": {"sha256": "<precomputed-sha256>"}
    }
  ]
}
```

```text
python3 experience_docx/tools/build_route_asset_manifest.py \
  --repo . --operation A0 --request /tmp/a0-assets.json --apply
```

Only `unrestricted` and `engineering_debug` assets may use `local_file` or
`local_checkout`; all development/confirmation/canary/sealed assets require a
predeclared SHA/commit and are not opened. The builder performs no discovery,
validates protected-data permissions against the existing runtime spec, and
refuses to replace an existing manifest.

## Reuse Real-Structure Fixture Assertions

Model routes may import `route_engineering_fixture.py` from their CPU contract
instead of reimplementing scope/no-op/finite/gradient/microfit assertions:

```python
from route_engineering_fixture import (
    assert_finite_tensors,
    assert_loss_decreased,
    assert_nonzero_gradients,
    assert_noop,
    assert_trainable_scope,
)
```

The route still constructs and executes the exact production model path. The
helper does not choose shapes, load rules, parameters, optimizer, loss, or
thresholds. Metadata-only routes do not import it.

## Validate One Compact Evidence Sync

In the clean evidence worktree, stage the complete compact bundle and run one
gate:

```text
git add <route-card-and-compact-evidence>
python3 experience_docx/tools/validate_evidence_sync.py \
  --repo . --route-id <route-id> --report /tmp/evidence-sync.json
```

The default requires `HEAD` to equal current GitHub main, added/modified
allowlisted text files only, one route closeout, valid UTF-8/JSON/CSV, matching
route identity, and no central index/family update. Add
`--allow-project-memory-update` only for a terminal route decision or explicit
major handoff. Add `--engineering-archive` only after the user explicitly chose
archive; it cannot be combined with project-memory updates. Normal repair never
uses that flag and never syncs the superseded failure bundle.

After `EVIDENCE_SYNC_OK`, commit and push once. Do not separately repeat diff,
size, JSON/CSV, suffix, or code-path checks already owned by this gate.
