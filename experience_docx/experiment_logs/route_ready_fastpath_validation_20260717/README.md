# Route-Ready Fastpath Validation Evidence

Status: engineering repair r2 planned after the archived r1 evidence-path
failure.

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

Route card:
`experience_docx/experiment_cards/2026-07-17-route-ready-fastpath-validation.md`.
