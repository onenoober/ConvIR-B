# convir-ops v4.3.1 Migrated Repair Reopen

Date: 2026-07-17

Status: `CLOUD_ACCEPTANCE_PASS`

Candidate commit `ea12f8aeaa5317831ad0fa228f470e1d0069a993` adds one
compatibility transition: an engineering receipt automatically archived by the
v4.3 signed-state migration may accept a later explicit user `repair` choice.
A normal explicit archive remains terminal.

The candidate preserves schema v4, exactly six tools, receipt HMAC integrity,
the evidence lock, and the rule that repair authorizes preparation only. Cloud
acceptance on `convir-4090` passed 81 tests in 3.262 seconds plus a fresh stdio
version/source/schema probe. No GPU, model, dataset, checkpoint, confirmation,
canary, locked test, route output, or production MCP state was accessed.

Primary evidence: `acceptance_summary.json`.
