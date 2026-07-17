# Route-Ready Fastpath Validation

Date: 2026-07-17

Status: COMPLETED

## Identity

- Route id: route_ready_fastpath_validation_20260717
- Question: Can one staged-snapshot validator and one generic lifecycle runner prevent repeated launch-packaging failures?
- Rules commit: ee2f8e4259c64eef83a1e93cd2535310fd86e92b
- Source branch/commit: github/main@ee2f8e4259c64eef83a1e93cd2535310fd86e92b
- Route branch: codex/route-ready-runner-v1-20260717
- Locked test/canary policy: prohibited; no model, GPU, dataset, checkpoint, confirmation, canary, or locked test is used

## Scientific Contract

- Population and analysis/grouping unit: synthetic route manifests, contexts, outputs, results, and evidence files
- Intervention or factor contrast and reference: generic route-ready/lifecycle protocol versus route-specific shell and positional-path packaging
- Primary outcome, direction and aggregation: all deterministic cloud unit and receipt-bound E2E checks pass
- Preferred mechanism and strongest competing explanation: shared schemas and context paths remove drift; a superficial static test could miss the real MCP lifecycle
- Evidence roles and candidate/freeze point: engineering_debug only; code, schemas, tests, and E2E tuple freeze in this commit
- Primary gate, uncertainty and threshold source: exact binary gate requiring all cloud tests, contract-before-run, evidence copy, and typed closeout validation
- `PASS` authorizes: ROUTE_READY_FASTPATH_ADOPTION only
- `INCONCLUSIVE` authorizes: one same-scope engineering review without model work
- `FAIL` stops: integration into main and use by model routes

## Implementation Contract

- Exact change and disabled mechanisms: add shared staged validator, lifecycle runner, program API, schemas, and tests; disable arbitrary shell commands and route-specific closeout ownership
- Checkpoint/load/init/freeze contract: not applicable; no checkpoint or model is loaded
- Input whitelist and prohibited inputs: synthetic manifests/contexts/files only; datasets, metrics, predictions, weights, images, GPU, and protected data are prohibited
- Dataset/split/preprocessing/metric identities: no dataset and no scientific metric
- Matched baseline and budget: CPU-only unit tests plus one synthetic contract/run lifecycle
- Resource/cost limits or descriptive-only rationale: under ten minutes, no GPU, one fresh output, compact text evidence only
- Runner and required assets: experience_docx/tools/run_route_operation.sh; no external runtime asset
- Runtime spec and `contract --context` / `run --context` entrypoint: `ROUTE_READY_FASTPATH_VALIDATION.json` and `route_ready_fastpath_validation.py`

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| ROUTE_READY_FASTPATH_VALIDATION | engineering_debug synthetic cloud validation | all unit and registered lifecycle checks pass | route-ready fastpath adoption |

- First operation: ROUTE_READY_FASTPATH_VALIDATION
- Expected wall time and monitor profile: under ten minutes; short
- Complete-unit resume policy: none; fresh output only
- Cloud workspace/run/output/status/closeout: schema-v4 derived fresh paths with generic control/contract/workload layout
- Compact Git evidence and cloud-only raw artifacts: summary and typed closeout only; unit-test stdout remains cloud-only

## Engineering Repair History

- r1: `FAILED_ENGINEERING / null / NONE` after contract pass; run output used
  `run/` while the frozen evidence contract required `workload/`. No model,
  GPU, dataset, checkpoint, confirmation, canary, or locked test was touched.
- r2: same scope/gate with only the directory mapping and a cross-component
  layout assertion repaired; new output, closeout, and summary names. It passed
  `COMPLETED_GATE_PASS / ROUTE_READY_FASTPATH_VALIDATION_PASS /
  ROUTE_READY_FASTPATH_ADOPTION` at commit
  `528ad61112a9cf7142a90864264f0230f18b93a3`.

## Authoring Fast-Path Acceptance

The later preparation-flow slimming update passed 83/83 CPU-only cloud tests
at main commit `2980c7970604c22a85242ca3ec669b030b08690b`. The accepted change
removes duplicate authoring checks, aggregates common errors, limits default
materialization to the currently authorized operation, defines representative
fixture reuse, and keeps repair-selected engineering evidence cloud-only. MCP,
schema, generic runtime files, scientific contracts, and protected-data rules
are unchanged. This acceptance used zero model calls, GPU access, and protected
data access.
