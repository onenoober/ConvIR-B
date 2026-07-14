# Unknown-Host And Whole-Task Routing Evaluation

Date: 2026-07-14

Status: `SCHEMA_V2_THREE_ENDPOINT_RUNTIME_VALIDATED`.

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

## Static Checks

The user explicitly authorized local dispatcher runtime validation for this
task. PowerShell parsing, JSON parsing, Bash syntax, `git diff --check`, and the
zero-model-call dispatcher regression set passed. The regression set covered
15 cases: R0/Luna, R1/Luna, R2/Terra, R3/Sol, unknown-host classified routing,
unknown-host typed routing, same-role routing, and five fail-closed cases.
Historical schema-v1 results under `model_agent_dispatcher/20260713/` remain
valid only for the exact candidates and hashes they record.

Static candidate hashes:

| File | SHA-256 |
| --- | --- |
| `tools/dispatch_agent_task.ps1` | `28f3baddcaa75e7d93e9b805d43eccc7e3834347389c91864ea5e0966bf05beb` |
| `tools/agent_model_dispatch_request.schema.json` | `b2ac980c3e19c2d2c3d3a95b322f7474575c7b3ab55a39e8af3cd2ae00c7197a` |
| `tools/test_dispatch_agent_task.ps1` | `16d233db505cc9a6b460e37fa72902f67599450ff586a251a2365786adefee9e` |

The updated deterministic test script includes positive cases for unknown-host
classified R0/R1/R2/R3 dispatch, unknown-host typed R0 dispatch, same-role
routing, and a route-specific R1 tuple. It keeps fail-closed coverage for a
stale rules commit, incomplete R1 checks, a target role below the class floor,
a typed handoff without a reference, and a mismatched source identity tuple.

## Three-Endpoint Runtime Validation

Runtime rules commit:
`7df7ae151911e8de077849215fdec8383c0b4dba`.

Tracked cloud execution commit:
`6077d258d9308e05310bc8225f57a820ed1a2223`.

Compact evidence is stored under
`experiment_logs/schema_v2_three_endpoint_dispatch_validation_20260714/`.
The validation used an isolated cloud repository, run root, and tmux session;
it requested no GPU and accessed no dataset, checkpoint, or existing experiment.

Observed route chain:

- The first Terra attempt exposed two real defects: WSL was blocked by the
  child sandbox, and a normally completed child turn could be mislabeled PASS
  without completing its requested action. No cloud write occurred.
- Schema v2 now requires an explicit execution scope and exact final-answer
  completion marker. Only `wsl_cloud_transport` maps to
  `danger-full-access`, and only when `next_action` names the tracked wrapper or
  `convir_route_*` MCP operations. Missing completion, tool-before-ack,
  escalation, or nonzero child exit fails the dispatcher.
- The corrected Terra R2/medium dispatch passed, created the isolated cloud
  checkout, and verified the GitHub and cloud commit.
- Luna R1/medium verified the GitHub authorization tuple and completed MCP
  preflight, launch, and the cloud probe. Its dispatcher result correctly
  failed closed because the request incorrectly required a stdout marker that
  the bounded monitor contract does not expose; the cloud closeout itself was
  already terminal PASS.
- A corrected Luna R0/low request used one bounded monitor call and passed on
  the typed status, route id, execution commit, closeout name, and MCP marker.
- A fresh Sol R3/high major handoff independently compared the GitHub evidence
  and live cloud terminal state, explicitly accounted for both failed attempts,
  and returned PASS with `SCHEMA_V2_SOL_REVIEW_OK`.

Cloud closeout state is `COMPLETED_VALIDATION_PASS`; probe and closeout agree on
schema version 2 and execution commit. Their safety fields record
`gpu_used=false`, `dataset_accessed=false`, and `experiment_modified=false`.
Temporary requests, raw dispatcher events, the launch authorization, tracked
probe runner, and isolated cloud workspace were removed after the compact
evidence and Sol verdict were archived. Cleanup verified the exact isolated
paths and emitted `SCHEMA_V2_CLOUD_CLEANUP_OK`; these assets are not durable
schema-v2 dependencies.

Decision:
`KEEP_SCHEMA_V2_CANDIDATE_THREE_ENDPOINT_VALIDATED`.
