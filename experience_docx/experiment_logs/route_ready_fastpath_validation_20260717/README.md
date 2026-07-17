# Route-Ready Fastpath Validation Evidence

Status: COMPLETED_GATE_PASS; route-ready generic runtime adoption authorized.

This CPU-only validation exercises the staged-snapshot validator, generic
lifecycle runner, contract-before-run entrypoint API, protected-data guards,
evidence copying, and receipt-bound typed closeout. It uses no model, GPU,
dataset, checkpoint, confirmation, canary, or locked test.

`route-ready-fastpath-validation-r1` reached contract pass and workload start,
but closed `FAILED_ENGINEERING / null / NONE`: the run context wrote under
`run/` while the frozen publisher required `workload/summary.json`. The typed
closeout is `route_ready_fastpath_validation_closeout.json` with SHA-256
`5de22fa5ecfcadd95aae88eefe39a119f215d4ac2020fe844d24b815696c3e23`.
No summary was published and no adoption was authorized. r2 changes only that
directory mapping plus an explicit cross-component layout assertion and uses
new output/evidence identities.

r2 passed the same frozen CPU-only E2E at route commit
`528ad61112a9cf7142a90864264f0230f18b93a3`. The terminal tuple is
`COMPLETED_GATE_PASS / ROUTE_READY_FASTPATH_VALIDATION_PASS /
ROUTE_READY_FASTPATH_ADOPTION`. The summary SHA-256 is
`107d3c19f9e53c7b5f3ce0049189144da2ccfa4c2a2f45c38b62c573cea94d38`;
the closeout SHA-256 is
`f4e0653f40d76c070fc23c6eb20156140f4dfd425cbdb035cec39ff094de3798`.
It records contract-before-run, generic runner use, position-free context paths,
typed lifecycle closeout, zero model calls, and no GPU, dataset, checkpoint,
confirmation, canary, or locked-test access.

The r1 launch crossed the old v4.1 timeout boundary after completing. One
bounded metadata-only inspection proved exact repo/runner/output/closeout and
the cloud-tested v4.2 candidate recovered its receipt without a second launch.
Candidate `980821176f09514d913f4ad0507e494b3c45971b` then passed 73 cloud
tests with schema v4 and exactly six tools; see
`convir_ops_4_2_candidate_validation.json`. Registered-service activation still
requires a fresh MCP process loading main.

After main integration at
`baced3179770f44b5a6ed750b1d3d585d9859adc`, an ordinary minimal route passed
the staged validator without bootstrap. The report records
`bootstrap_missing_from_main=[]` and `runtime_bundle_canonical=true`; no cloud
operation was launched. See `strict_main_validation.json` and strict-check
branch commit `03d24d77487410aa35ea90b6ecd08415a2ef39bf`.

The 2026-07-17 route-authoring slimming update passed its final CPU-only cloud
acceptance at `2980c7970604c22a85242ca3ec669b030b08690b`: 83/83 tool tests,
targeted Python compilation, and diff checks passed. The MCP server and generic
runtime bundle are byte-unchanged from baseline main
`dd17b42237bee1cc2663a75b91bdc1dd85c48f74`; model calls, GPU access, and
protected-data access were all zero. See
`route_authoring_fastpath_cloud_acceptance.json`.

Route card:
`experience_docx/experiment_cards/2026-07-17-route-ready-fastpath-validation.md`.
