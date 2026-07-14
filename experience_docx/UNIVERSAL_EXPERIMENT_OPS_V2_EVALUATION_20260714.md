# Universal Experiment Operations V2 Evaluation

Date: 2026-07-14

Status: L5 workflow-change audit; adoption gate passed.

## Scope And Baseline

This evaluation covers the general experiment workflow, deterministic model
dispatcher schema v2, and `convir-ops-v2`. It is not an optimization for one
historical long run.

- rules baseline: GitHub `main@df1c03672fb9c3b45a82f287fa2cb99b51b2c1cc`;
- evaluated branch: `codex/universal-experiment-ops-v2-20260714`;
- first complete workflow/tool commit: `954d9e0e2add07fe644154db537b0e2473d4eea5`;
- cloud-tested dispatcher commit: `f77e4a8b1d4abc4bdc14471eb1cb56481c5d38ac`;
- runtime host: `convir-4090`;
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`,
  version 3.10.20.

No GPU job, dataset, checkpoint, model inference, canary, or locked test was
accessed during this workflow evaluation.

## General Workflow Finding

The long workflow contained avoidable orchestration work, but its scientific
gates were not the redundancy. The removable costs were repeated task startup,
re-reading a warm route contract, separate smoke launches, model-visible poll
loops, repeated transport construction, and repeated lifecycle parameter
generation.

The adopted route is the fewest safe decision packages:

1. one R3 scientific design package;
2. one R2 implementation and engineering-repair package;
3. one R1 authorized stage package, with runner-integrated pre-smoke;
4. at most one persistent Luna observation task for a healthy long run;
5. one R3 terminal interpretation package and only a materially amortized R2
   archive package.

These packages do not replace experiment stages. Stage count remains determined
by the estimand, evidence roles, independence requirements, adaptive contract,
and written decision value. Preflight, locked-test policy, metric alignment,
fresh workspace, typed authorization, resource checks, and closeout provenance
remain mandatory.

## Adopted Optimizations

| Area | Decision | Reason |
| --- | --- | --- |
| Sequential experiment design | Adopt | Runs the cheapest decision-changing block first while preserving preregistered branches, interaction controls, and independent confirmation. |
| Integrated pre-smoke | Adopt | Checks identity, native shape/batch, no-op, finite values, and tiny update inside the tracked runner before expensive work, eliminating a separate launch boundary. |
| Wall-time and heartbeat contract | Adopt | Makes stalls distinguishable from slow valid phases and bounds model-visible observation. |
| Persistent monitoring task | Adopt | Removes child-per-poll context reload without allowing the monitor to interpret results or repair code. |
| `convir-ops-v2` six-tool surface | Adopt | Normal lifecycle is only `plan_manifest -> start_authorized -> finish`; recovery primitives are no longer model-visible. |
| Dispatcher R1 evidence binding | Adopt | Dispatcher independently compares the GitHub JSON `route_id/state/decision/authorizes` tuple instead of trusting a caller boolean. |
| Schema-v1 compatibility surface | Remove from active use | Keeping old lifecycle names callable would make routing ambiguous and retain unnecessary tool context. Historical evidence remains readable. |

## Reliability Fixes Required Before Adoption

The candidate was not adopted unchanged. This evaluation required the following
fixes:

- batch the route-head and current-main `ls-remote` query;
- update `github/main` with an explicit remote-tracking refspec after a
  single-branch clone;
- seal GPU free-memory/utilization thresholds and recheck the same GPU at launch;
- use `flock` for persistent plan/receipt records;
- treat a missing heartbeat from an active session as stale after the sealed
  timeout;
- treat an ended session without closeout as `CLOSEOUT_MISSING`, not as an
  indefinitely pollable state;
- require closeout `route_id`, `run_id`, `route_commit`, `runner_sha256`, and
  allowed terminal tuple to match the receipt;
- recheck the real closeout path at launch rather than a placeholder;
- require every R1 dispatcher request to bind a typed GitHub JSON handoff;
- reject schema-v1 lifecycle tool names before dispatch.

## Validation Evidence

Cloud validation at the exact candidate commits produced:

- `convir-ops-v2` mocked lifecycle/MCP: `21/21 PASS`;
- model-visible MCP surface: exactly 6 tools;
- dispatcher schema-v2 dry-run matrix: `20/20 PASS`;
- dispatcher model calls: `0`;
- R0/R1 -> Luna, R2 -> Terra, R3 -> Sol routing: pass;
- stale rules, incomplete or mismatched R1 evidence, unbound R1, under-role R2,
  invalid transport contract, identity mismatch, and legacy lifecycle names:
  all fail closed;
- local permitted checks: Python compile, PowerShell parser, JSON parser, and
  `git diff --check` pass.

The dispatcher matrix ran on `convir-4090` with a temporary PowerShell Core
7.4.6 extraction under `/tmp`; it made no model calls and left no installation
or validation checkout behind.

## Efficiency Evidence And Limits

The following savings are directly established by interface structure:

- model-visible MCP tools decrease from 10 to 6 (`40%`);
- the fixed manifest lifecycle retains the previously measured `42%` argument
  reduction;
- manifest planning no longer performs cloud SSH reads;
- start composes prepare and launch while preserving two dynamic checks;
- finish combines a server-side observation window and closeout validation in
  one SSH call;
- one monitoring task can reuse the receipt for all healthy windows.

Actual end-to-end token, credit, and wall-time savings are not yet proven for a
representative set of real routes. They must be measured without rerunning an
experiment solely for cost accounting. New routes record task count,
model-visible observation count, phase wall time, dispatcher startup time, and
repair/escalation count. Promotion does not depend on claiming an unmeasured
percentage.

## Decision

`ADOPT_UNIVERSAL_EXPERIMENT_OPS_V2`.

Adoption is justified by passed safety/behavior matrices, removal of known
failure loops, a smaller external contract, and preserved scientific gates.
The old MCP and schema-v1 dispatcher must be deactivated after the v2 source is
present in a clean `operations-v2` worktree. Historical branches and evidence
may remain, but no active configuration or current rule may route to v1.
