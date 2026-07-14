# Unknown-Host And Whole-Task Routing Evaluation

Date: 2026-07-14

Status: `STATIC_IMPLEMENTATION_COMPLETE_RUNTIME_VALIDATION_PENDING`.

This L5 audit supports `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`. It is not an
execution protocol, model qualification result, experiment authorization, or
replacement for the dated dispatcher runtime evidence.

## Problems Evaluated

The prior workflow coupled target-model dispatch to knowledge of the current
host role. That forced an interactive selector confirmation even though the
dispatcher starts a new task with an explicit model and effort. It also
optimized individual switches rather than the remaining task envelope, which
could spend more uncached tokens on repeated cold starts.

Two implementation defects were confirmed in the schema-v1 dispatcher:

- WSL git stderr was merged into stdout before SHA parsing, so a successful WSL
  warning could corrupt a rules or route SHA;
- R1 execution was restricted to generic `PASS` / `CONTINUE` values instead of
  accepting an exactly verified route-specific
  `route_id` / `state` / `decision` / `authorizes` tuple.

## Candidate Change

The canonical L2 protocol and schema-v2 candidate now separate source identity
from target qualification. The candidate supports an unknown orchestration
host and a task-scoped user pin. Any host may act as the scheduler, mechanically
apply the canonical task-class table, and route down, up, or laterally. Source
identity is audit-only; ambiguous classification fails closed to R3/frontier.

The dispatcher candidate:

- records source identity, role, effort, routing basis, and basis reference;
- requires a typed basis reference to name a remotely fetchable GitHub commit
  and an existing path within that commit;
- requires unknown identity/role/effort to agree;
- does not compare source and target ranks or require host identity for routing;
- accepts `dispatcher_classification` from known, unknown, or task-pinned hosts;
- relies on the canonical hard assignments to send any ambiguous class to
  `R3` / `frontier` / `high`;
- preserves class floors, model qualification, route-commit checks, exact R1
  checked-field coverage, child marker acknowledgement, and tool-before-ack
  rejection;
- parses WSL git stdout and stderr separately; and
- treats an exactly checked non-`NONE` route-specific `authorizes` value as the
  R1 execution grant instead of interpreting route-specific state names.

Current routing rules now minimize the remaining task's total uncached input,
output/reasoning tokens, and official credit-equivalent cost. They prefer one
durable frontier plan, one engineering batch, amortized bounded-operation
batches, and a compact frontier interpretation handoff over a new child for
every adjacent operation.

## Static Validation Boundary

Local policy permits editing and syntax/static checks only. No dispatcher dry
run, child model call, cloud command, experiment, evaluation, or inference was
executed for this change. Historical schema-v1 results under
`model_agent_dispatcher/20260713/` remain valid only for the exact candidates
and hashes they record.

Static candidate hashes:

| File | SHA-256 |
| --- | --- |
| `tools/dispatch_agent_task.ps1` | `2a0f04ab17cef8245e15c1e84cb60c35fc4fe19ce8c614a21380727764fe118c` |
| `tools/agent_model_dispatch_request.schema.json` | `d19a0e68298cb90d3b2d552769b03faf46183a303b3bc6dd5129ef34652c1a1a` |
| `tools/test_dispatch_agent_task.ps1` | `974ce3e0ed6d73a51731184519cb139f69a3e3f45a165812a0e900404bacab48` |

The updated deterministic test script adds positive cases for unknown-host
classified R0/R1/R2/R3 dispatch, unknown-host typed R0 dispatch, same-role routing,
and a route-specific R1 tuple. It keeps fail-closed coverage for a stale rules
commit, incomplete R1 checks, a target role below the class floor, a typed
handoff without a reference, and a mismatched source identity tuple. Before
promotion to GitHub `main`, run the static syntax checks and the dated
dispatcher dry-run/end-to-end validation on an explicitly authorized runtime,
then archive the candidate hashes and compact results in a new dated validation
record.

Decision:
`KEEP_SCHEMA_V2_CANDIDATE_FAIL_CLOSED_PENDING_RUNTIME_VALIDATION`.
