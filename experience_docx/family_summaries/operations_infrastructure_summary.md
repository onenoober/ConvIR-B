# Operations Infrastructure Summary

Date: 2026-07-17

Status: generic run monitoring and the route-ready generic runtime are adopted;
schema-v4 start-recovery v4.2 passed candidate tests, is on main, and awaits
fresh-process activation.

## Current Verdict

The first two cloud candidates stopped on validation-harness defects and were
archived with `FAILED_ENGINEERING` closeouts. The strengthened candidate passed
seven telemetry tests, 22 restricted control-plane tests, semantic control/data-
read rejection, atomicity, fail-open behavior, exact-parent exit, stale/closeout
recovery, bounded finish, and the cost/file gate. The generic subset then
reached GitHub main and the registered-service E2E passed `convir-ops 4.1.0`,
exactly six schema-v4 tools, receipt-bound closeout/evidence, and the fixed
cost/safety contract. The terminal closeout authorizes generic monitoring
adoption and confirms zero model calls and no GPU, dataset, checkpoint, canary,
or locked-test access.

The route-ready fast path then passed its r2 CPU-only receipt-bound E2E at
`528ad61112a9cf7142a90864264f0230f18b93a3`. It validates one staged snapshot
with the exact MCP parser, uses one unchanged generic runner plus declarative
runtime specs and context-only Python entrypoints, publishes evidence/closeout
write-once, and records zero model/data/GPU access. r1 had exposed and archived
a `run/` versus `workload/` integration defect before r2 repaired only that
mapping.

r1 also exposed an old v4.1 control-plane dead end: start timed out after a
valid closeout but had no receipt. The v4.2 candidate keeps exactly six
schema-v4 tools, adds one same-plan metadata-only receipt recovery, and uses the
immutable cloud anchor as a shared Git object seed. Candidate
`980821176f09514d913f4ad0507e494b3c45971b` passed 73 cloud tests with zero
model calls. It is not registered-service evidence until a fresh MCP process
loads v4.2 from main.

## Revalidation Condition

Use the adopted protocol for future long operations without a resident model or
polling watcher. Revalidate before use only if the telemetry payload/permissions,
process-observation mechanism, six-tool schema, remote transport, stale/finish
semantics, or cost bound changes. Scientific route authorization remains
independent and must still come from each route's own typed closeout.

Use `ROUTE_READY_FASTPATH.md` for new routes. Revalidate the generic runtime
only when its context/runtime/asset/evidence schemas or lifecycle ownership
change; ordinary route entrypoints require only their one staged gate and cloud
contract.
