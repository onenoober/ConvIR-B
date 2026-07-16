# Operations Infrastructure Summary

Date: 2026-07-16

Status: generic run monitoring adopted after candidate and receipt-bound E2E
gates passed.

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

## Revalidation Condition

Use the adopted protocol for future long operations without a resident model or
polling watcher. Revalidate before use only if the telemetry payload/permissions,
process-observation mechanism, six-tool schema, remote transport, stale/finish
semantics, or cost bound changes. Scientific route authorization remains
independent and must still come from each route's own typed closeout.
