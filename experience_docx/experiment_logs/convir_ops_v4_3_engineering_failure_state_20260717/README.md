# convir-ops v4.3 Engineering Failure State

Date: 2026-07-17

Status: adopted after cloud acceptance, fresh-process activation and signed
production-state migration.

Candidate commit: `e2b1c15906163ad7f587d8daa9581854426a1c15`.

The candidate retains schema v4 and exactly six tools while adding a
receipt-bound engineering-failure decision gate. A validated
`FAILED_ENGINEERING / null / NONE` closeout now enters
`ENGINEERING_REVIEW_REQUIRED`; evidence access remains locked until the user
explicitly selects `repair` or `archive` through the same finish tool.

Cloud acceptance on `convir-4090` used the fixed project Python and a temporary
Git checkout of the exact candidate branch. The final candidate with the state
migration test passed all 79 tests in 3.252 seconds. A separate stdio probe verified server `4.3.0`, six tools, the
`repair|archive` finish enum, and source SHA-256
`59adcfa653646d3f0b6d10afe5f1b72171f163d214e2c07f0884a974e7bc4e66`.

Production activation updated the dedicated configured worktree to
`github/main@a42c46e61bc1b66df7470377991d8bdc8a27f383`. A fresh stdio process
returned the same version, source SHA and six-tool schema. Signed state
migration removed 12 expired plan records, retained all eight historical
receipts and the 32-byte HMAC key, migrated four already-archived engineering
receipts and three scientific receipts, and preserved one closeout-missing
receipt unchanged. The obsolete isolated v4.2 E2E state directory was removed;
no cloud output, closeout, production receipt, route worktree or experiment
evidence was deleted.

No model, GPU, dataset, checkpoint, confirmation, canary, locked test, route
output, or production MCP state was accessed by the acceptance script.

Primary evidence: `acceptance_summary.json`.
