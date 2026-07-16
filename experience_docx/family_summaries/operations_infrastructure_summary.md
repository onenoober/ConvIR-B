# Operations Infrastructure Summary

Date: 2026-07-16

Status: generic run monitoring remains validation-only.

## Current Verdict

The metadata-only monitoring design remains validation-only. The first cloud
candidate stopped because a raw-text audit matched `signal.` in a docstring.
The semantic replacement correctly ignored prose and added negative controls,
but the second candidate stopped because its method-name extraction skipped a
compound `Path("/proc") / ...` receiver. Five of six telemetry tests passed in
each candidate. Both closeouts confirm zero model calls and no GPU, dataset,
checkpoint, canary, or locked-test access.

## Reopen Condition

Correct only the syntax-tree method matcher so it directly inspects attribute
calls with compound receivers while retaining the exact
`/proc/<pid>/stat` allowlist and all negative controls. Re-run the same frozen
CPU-only cases under a new candidate commit and output identity. Do not adopt,
integrate to `main`, reload the MCP, or use monitoring in a model experiment
until telemetry, transport, bounded finish, cost, and typed-closeout gates all
pass on `convir-4090`.
