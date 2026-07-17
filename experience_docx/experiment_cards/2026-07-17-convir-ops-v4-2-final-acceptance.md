# Convir Ops v4.2 Final Acceptance

Date: 2026-07-17

Status: `COMPLETED_PASS`

## Identity

- Route id: `convir_ops_v4_2_final_acceptance_20260717`.
- Question: Does the restarted registered service load the tested generic v4.2
  control plane and remain deterministic, non-mutating, and bounded before
  future scientific routes use it?
- GitHub main at acceptance:
  `da4effe39ae32fdb815cbb9fbfb3372ec41ac251`.
- Cloud validation branch/commit:
  `codex/route-ready-runner-v1-20260717@8a93cbb9af9c9731e2fe118cfb241edaf22067cb`.
- Evidence role: engineering infrastructure acceptance only.
- Protected-resource policy: no model, GPU, dataset, checkpoint, confirmation,
  canary, or locked-test access.

## Frozen Gate

Pass requires all of the following:

1. the unchanged tracked cloud validator exits zero with an explicit success
   marker, reports server `4.2.0`, schema v4, exactly six tools, all tests
   passing, and zero model calls;
2. the fresh registered process source SHA-256 equals the GitHub-main source;
3. three consecutive live `convir_git_status` calls return the same clean and
   fresh main identity with no route evidence changes or Git mutation; and
4. no plan, start, finish, model operation, or protected-data access occurs in
   this activation audit.

Any mismatch is a failure and blocks adoption. Historical receipt-bound
plan/start/finish/evidence E2E evidence remains authoritative for the launch
lifecycle; this audit specifically closes the fresh-process v4.2 activation
condition.

## Result

- Cloud validator transport: `CONVIRCTL_REMOTE_SCRIPT_OK`, exit code `0`.
- Cloud gate: `CONVIR_OPS_4_2_VALIDATION_OK`.
- Cloud tests: `73/73`; tools: `6`; model calls: `0`.
- Tracked validator SHA-256:
  `a8f3a198dfbb2140cbe2c72f5d6b619e24fdd0ae778bebea363fee4de1a903c8`.
- Registered executable:
  `/home/ubuntu/workspace/ConvIR-B-operations-v4/experience_docx/tools/convir_ops_mcp.py`.
- Registered and main source SHA-256:
  `f84330ffc1ffe5b6973f710078e81bfb35bd4ffccab97e15096397e6e75d6e8a`.
- Live read-only audit calls: `3/3` identical; clean worktree, fresh GitHub main,
  empty changed/evidence paths, both diff checks passing, and
  `git_mutations_performed=false`.
- No experiment plan or cloud model run was created.

## Decision

Verdict: `PASS`.

Decision: `CONVIR_OPS_V4_2_ADOPTION`.

This authorizes the unchanged generic operations layer for later route-ready
experiments. It does not authorize A1X, training, evaluation, inference,
confirmation, canary, or locked-test access. Route science still requires its
own frozen card, staged gate, exact commit, CPU contract, and typed closeout.

The registered path is currently a clean route-ready worktree whose MCP source
is byte-identical to main. That is valid for this pinned source. A future MCP
source, schema, transport, lifecycle, telemetry-permission, or finish-semantics
change requires a clean dedicated registration update, process restart, and
revalidation. `exact_resume` remains unsupported; recovery uses a new output
and, when declared, verified complete-unit assets.
