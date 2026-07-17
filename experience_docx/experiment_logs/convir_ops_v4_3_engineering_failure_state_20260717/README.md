# convir-ops v4.3 Engineering Failure State

Date: 2026-07-17

Status: cloud acceptance passed; production activation pending main integration.

Candidate commit: `e2b1c15906163ad7f587d8daa9581854426a1c15`.

The candidate retains schema v4 and exactly six tools while adding a
receipt-bound engineering-failure decision gate. A validated
`FAILED_ENGINEERING / null / NONE` closeout now enters
`ENGINEERING_REVIEW_REQUIRED`; evidence access remains locked until the user
explicitly selects `repair` or `archive` through the same finish tool.

Cloud acceptance on `convir-4090` used the fixed project Python and a temporary
Git checkout of the exact candidate branch. All 78 tests passed in 3.139
seconds. A separate stdio probe verified server `4.3.0`, six tools, the
`repair|archive` finish enum, and source SHA-256
`59adcfa653646d3f0b6d10afe5f1b72171f163d214e2c07f0884a974e7bc4e66`.

No model, GPU, dataset, checkpoint, confirmation, canary, locked test, route
output, or production MCP state was accessed by the acceptance script.

Primary evidence: `acceptance_summary.json`.
