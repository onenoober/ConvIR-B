# Model-Agent Cost Routing Protocol

Date: 2026-07-14

Status: canonical L2 protocol for selecting the cheapest qualified agent model
without changing experiment semantics or weakening execution gates.

## Purpose And Authority

Use this protocol before experiment planning, implementation, cloud operation,
result interpretation, or evidence sync. It selects the agent model and
reasoning effort for the current task. It does not authorize a route, stage,
metric, command, locked-test action, or scientific continuation.

All experiment authority remains with current GitHub evidence, the route card,
the previous typed closeout, and the matching execution protocol. Changing the
agent model must never change the route commit, runner, data, checkpoint,
metric contract, gate, output root, or authorization chain.

Current model-role mapping, based on the public GPT-5.6 family on 2026-07-13:

| Stable role | Current model | Default effort | Meaning |
| --- | --- | --- | --- |
| `frontier` | GPT-5.6 Sol | high | Highest-risk scientific and semantic decisions. |
| `balanced` | GPT-5.6 Terra | medium | Existing-contract engineering and reviewed writes. |
| `fast` | GPT-5.6 Luna | low or medium | Bounded, deterministic, mostly read-only operations. |

The role is canonical; the model name is a dated mapping. When the model family
changes, update only this table after qualification. Do not rewrite route cards
or historical evidence merely because a newer model exists.

Current qualification state:

| Role/model | Status | Maximum current class |
| --- | --- | --- |
| `frontier` / GPT-5.6 Sol | strongest-model baseline | `R3` |
| `balanced` / GPT-5.6 Terra | qualified 2026-07-13 | `R2` |
| `fast` / GPT-5.6 Luna | qualified 2026-07-13 | `R1` |

Qualification evidence:
`model_agent_qualification/20260713_gpt56/README.md`. Both candidate models
scored `91/91` critical fields with zero unauthorized actions and zero observed
tool calls under `codex-cli 0.144.1`; the reviewer was `frontier` / GPT-5.6 Sol.

Change this table only with a dated L5 qualification result that records the
model id/version, case manifest, exact critical-field score, unauthorized-action
count, decision, and reviewer. The workflow evaluation file is not itself a
qualification pass.

## Host Identity Modes And Effort

Host identity is task configuration, not experiment evidence. It controls
whether the current host may execute work itself; it is not needed to select an
explicit child model. Use one of these modes:

| Mode | Source | Lifetime | Host behavior |
| --- | --- | --- | --- |
| `dispatcher_receipt` | verified child receipt | child task | Execute only the received class and bounded action. |
| `product_metadata` or `cli_status` | product-supplied metadata or current CLI `/status` | current task/session | The qualified host may execute or dispatch under the amortization rule. |
| `user_pinned_task` | user explicitly names the visible model and effort and states that they remain fixed | current interactive task | Reuse the pin across turns without asking again. |
| `unknown` | no trustworthy identity receipt | until a trustworthy source appears | Apply the routing table and dispatch; do not perform the routed experiment action itself. |

A `user_pinned_task` identity remains valid across ordinary turns and context
compaction in the same interactive task. Invalidate it on a new/forked task, an
explicit `/model` or `/reasoning` change, conflicting product metadata, loss of
task continuity, or user withdrawal. When it becomes invalid, fall back to
`unknown`; do not ask for confirmation merely to launch an explicit child.
The pin is never experiment evidence and is never propagated as a child
identity. A dispatched child uses its dispatcher receipt.

Host identity is audit data for model switching. It does not constrain whether
the dispatcher may route down, up, or laterally. Any host may perform the
minimal orchestration needed to classify the requested action with the
canonical table and create, dry-run, and execute a dispatcher request. An
unknown host must not perform the routed experiment action itself. Use one of
two routing bases:

1. `typed_handoff`: copy the exact class, role, authorization tuple, and bounded
   action from a durable GitHub handoff and cite it as
   `routing_basis_ref=github:<commit>:<path>`;
2. `dispatcher_classification`: apply the hard task-class assignments in this
   protocol. Clear bounded work may be routed to a lower qualified role
   regardless of the host model. Any ambiguity about scientific meaning,
   failure class, authorization, canary, or locked test is `R3` / `frontier` /
   `high`.

This classification grants no experiment authority to the host; the explicitly
selected child owns the bounded action. If the dispatcher is unavailable,
either establish a known qualified host for direct execution or stop. Manual
selector confirmation is a fallback for direct execution, not a prerequisite
for routing.

Shell environment variables are not an identity channel. Missing
`CODEX_MODEL`, `OPENAI_MODEL`, or similar variables are non-evidence. When two
known sources conflict, prefer a dispatcher receipt or current product metadata
over a user pin and invalidate the weaker source.

Reasoning effort is a minimum, not an equality check. The order is
`low < medium < high < xhigh`; `xhigh` therefore satisfies a `high` minimum. A
higher qualified model or effort may perform a lower-class task when keeping
the current context minimizes the remaining task's total token and credit cost.

Known tasks use the standard marker:

```text
MODEL_ROUTE class=<class> role=<stable-role> effort=<active-effort>
```

An unknown orchestration host may report:

```text
MODEL_ROUTE class=R0_READ_ONLY role=unknown effort=unknown
```

Dispatcher children must always use a known role and known effort. Record the
host identity mode in adjacent progress text or in the schema-v2 request; do not
add identity fields to the marker itself.

## Non-Negotiable Reliability Invariants

Model down-routing is allowed only while all of these remain unchanged:

- GitHub `main` or the named route branch is the experiment-memory source;
- local WSL remains editing and syntax/compile-only;
- cloud runtime remains `convir-4090` with the explicit cloud Python;
- every new route uses a fresh workspace and the authorized source commit;
- every launch has current dynamic preflight, a tracked runner, a fresh session
  and output directory, status/log capture, and a typed closeout;
- the previous closeout explicitly authorizes the launched stage;
- formal gates use their written metric contract and canonical Gate Policy;
- locked test remains blocked without explicit prior authorization;
- compact evidence selection, staging, push, and remote verification remain
  unchanged.

If reducing context or model tier would skip any invariant, do not down-route.

## Task Classes And Minimum Models

Classify the current task envelope before the first substantive tool call and
again at a durable boundary or when scope expands.

| Class | Typical work | Minimum model | Allowed behavior |
| --- | --- | --- | --- |
| `R0_READ_ONLY` | status, bounded monitor, exact-path read, SHA/list/tree inspection, compact metric extraction | qualified `fast`, low | No writes and no scientific interpretation beyond reporting typed fields. |
| `R1_BOUNDED_EXECUTION` | MCP preflight/launch of an already authorized tracked runner, explicit compact evidence fetch, syntax/static validation | qualified `fast`, medium | Inputs must come verbatim from the current card/closeout; no command, parameter, or route redesign. |
| `R2_ENGINEERING_CONTROL` | route setup from a written design, transport repair, typed-closeout integrity audit, explicit route-branch evidence commit, terminal sync with unchanged verdict | `balanced`, medium | May make reviewed engineering changes that do not alter scientific semantics; new commit/run id required when protocols require it. |
| `R3_SCIENTIFIC_AUTHORITY` | bottleneck synthesis, new route/gate/metric design, training or model-structure code, result interpretation, ambiguous failure classification, family verdict, promotion, canary or locked test | `frontier`, high | Owns scientific meaning and irreversible/high-impact decisions. |

Additional hard assignments:

- Any change to data, split, checkpoint, optimizer, loss, model structure,
  metric, threshold, gate, or authorized stage is `R3`.
- Any terminal scientific `PASS`, `FAIL`, `INCONCLUSIVE`, promote/stop/reopen
  decision is `R3`, even when a script computed the metrics.
- Locked-test selection, launch authorization, interpretation, and post-result
  restrictions are always `R3`.
- A routine GitHub `main` sync with no verdict change is `R2`; a family verdict,
  reopen condition, or current-route conclusion change is `R3`.
- A command/transport failure is at least `R2`; never let an `R0`/`R1` agent
  reinterpret partial output as scientific evidence.

## Task Envelope And Continuity

Classify the user-visible outcome and the highest-authority decision still
needed inside the current task, not each supporting tool call in isolation.
Exact-path reads, cloud inventory, or compact metric extraction performed to
support an unresolved `R3` interpretation remain inside that `R3` envelope.
They do not turn the active task into an independent `R0` task.

Raise the class immediately when new scope requires it. Lower the class only
after a durable boundary has closed the higher-class meaning and written the
minimum handoff payload. A lower-class continuation must have one bounded next
action that can finish without recovering the parent reasoning or making the
parent decision. This preserves both scientific ownership and context cost.

A stronger active role may complete adjacent lower-class operations in the same
envelope. After a durable boundary, however, eligible standalone repetition or
a bounded batch should use the dispatcher by default. Keeping that independent
work on a stronger model requires a concrete `dispatch=not_amortized` reason;
the user's stronger default model is not such a reason.

## Qualification Gate

`fast` may perform `R0` immediately. It may perform `R1` only after the current
fast model/version passes the repository qualification audit. `balanced` may
perform `R1`/`R2` only after its current model/version passes the same critical-
field audit at those scopes. Until then, use `frontier` for writes and decisions.

In addition to model qualification, a fast `R1` launch requires the tracked
runner or bounded preflight to machine-check the prior closeout's exact
`route_id`, `state`, `decision`, and `authorizes` values. A check of `state`
alone is insufficient. If any value is unstructured, implicit, or checked only
by model judgment, use at least a qualified `balanced` model and escalate any
ambiguity to `frontier`.

Qualification uses compact historical cases and requires exact agreement on:

- authoritative fact source;
- route id, branch, commit, stage, runner, and output root;
- `state`, `decision`, `authorizes`, and failure class;
- locked-test and canary policy;
- explicit evidence allowlist;
- whether to continue, stop, or escalate.

The acceptance line is `100%` on these safety-critical fields with zero
unauthorized writes or continuations. Quality wording and explanatory style are
not scored. Repeat qualification after a model-family/version change or after
one safety-critical execution error.

Until `fast` is qualified for `R1`, route bounded execution through a qualified
`balanced` model. Until `balanced` is qualified, use `frontier`. Qualification
never grants `R2` or `R3` authority to `fast`, or `R3` authority to `balanced`.

## Fail-Closed Switching

The active task cannot assume that a model switch happened. A known host whose
task class requires a higher role must:

1. stop before the next external write, launch, commit, push, or scientific
   decision;
2. emit `MODEL_SWITCH_REQUIRED` with the required stable role, current task
   class, exact durable handoff evidence, and the blocked next action;
3. resume in a new task or after an explicit model switch;
4. reread only the minimal durable handoff, not the prior chat transcript.

Model switching never compares the source role with the target role. The host
uses `task_routing` to launch the lowest qualified role for the declared class,
whether that is a downgrade, upgrade, or same-role child, and then stops owning
the dispatched action. Do not require the user to reveal or change the host
selector for this path. A known host role matters only when deciding whether the
current host may execute an action without dispatch.

Do not claim an automatic in-task switch without a verified product capability.
The repository dispatcher creates a new ephemeral Codex task; it does not
replace the model of the current task. A model may down-route a later
independent task, but it must not delegate high-risk meaning to a lower role
through an ordinary prompt or subagent.

## Deterministic External Dispatcher

Repository state: the known-source v1 behavior is validated in
`model_agent_dispatcher/20260713/README.md`. Schema-v2 unknown-host and
task-pinned behavior is implemented with a fail-closed static audit in
`MODEL_ROUTING_UNKNOWN_HOST_TOTAL_COST_EVALUATION_20260714.md`; run its dated
end-to-end validation before promoting that candidate to GitHub `main`.

Use `experience_docx/tools/dispatch_agent_task.ps1` when an explicit model-task
boundary passes the switching and context-amortization rules below. Its request
contract is owned by
`experience_docx/tools/agent_model_dispatch_request.schema.json`.

The dispatcher does not make an extra model call to choose a role. The host acts
as the scheduler: it copies a typed handoff or mechanically applies the task
class table, writes one schema-valid request, and stops that scope. Ambiguous
classification is always declared `R3`, not guessed downward.
The external process fetches `github/main`, rejects a stale rules commit, parses
the canonical role and qualification tables, validates the route worktree HEAD,
and starts exactly one ephemeral `codex exec` task with the selected model.
The explicit child `--model` and effort override the user's default model for
that child, so a Sol-default interactive task can still dispatch qualified work
to Terra or Luna.

Dispatch is allowed only for:

- source-independent `task_routing` to the qualified role for the declared
  class;
- standalone repeated or batched bounded work delegated to a cheaper role; or
- an explicit major handoff, including a same-role fresh task.

Do not dispatch one adjacent short operation merely because a cheaper model is
qualified when a known qualified host will continue it more cheaply in the
warm context. A pure scheduling host instead batches adjacent operations with
the same class, route commit, authorization, and context into one bounded child
request. The schema-v2 request records source identity/role/effort for audit,
but only target class/role/effort, qualification, and authorization affect
dispatch acceptance.

Conversely, do not keep an independent lower-class scope on the user's default
model merely for convenience. Repeated/long monitoring, a standalone bounded
status or evidence batch, and written-design-to-engineering handoffs are the
intended cost-saving boundaries. The dispatcher validation records successful
explicit selection of Luna for `R0/R1`, Terra for `R2`, and Sol for `R3`.

For `typed_handoff`, `routing_basis_ref` must use
`github:<commit>:<path>`; the dispatcher fetches the commit from `github` and
fails if the path does not exist. `dispatcher_classification` uses no reference.
The dispatcher does not enforce source/target rank direction. It enforces the
declared class floor, dated target qualification, target effort, route identity,
and authorization contract. A host may therefore route to a cheaper, stronger,
or same role without exposing its own identity.

Before any tool call, the child task must emit the exact `MODEL_ROUTE` marker
and acknowledge the dispatcher handoff SHA. The dispatcher fails if the marker
is absent, the SHA is absent, a tool starts before acknowledgement, the model
is unqualified, the route commit differs, or a fast `R1` request lacks a
machine-verified `route_id`/`state`/`decision`/`authorizes` tuple. It never
bypasses the experiment authorization, cloud preflight, metric, gate, locked-
test, or evidence-sync protocols.

The default invocation is a zero-model-call dry run. Add `-Execute` only after
the request is complete and the next action is already authorized. Store raw
dispatcher events outside the repository and archive only a compact terminal
audit when routing behavior or qualification changes.

### ConvIR Experiment Boundary Recipe

For every new ConvIR experiment route, apply this finite sequence. This section
owns the sequence; route cards and other protocols should link here instead of
copying it.

| Boundary | Task class / role | Dispatcher action |
| --- | --- | --- |
| Unclassified or ambiguous large task | `R3` / `frontier` | Use `task_routing`; require one frontier planning child to write a compact whole-task routing plan before lower-role work. |
| Bottleneck synthesis, route question, gate or metric design | `R3` / `frontier` | Route to frontier independently of source identity, or remain only when a known frontier host is intentionally executing the warm envelope. |
| Written design to fresh workspace, tracked runner, engineering repair | `R2` / `balanced` | Dispatch one standalone or batched engineering scope when it amortizes context reload. |
| Exact-tuple preflight, authorized launch, repeated monitor, explicit evidence fetch | `R1` / `fast` | Dispatch a bounded batch only after `route_id`, `state`, `decision`, and `authorizes` are machine-verified. Keep a short adjacent preflight/launch/monitor sequence in the current qualified balanced task. |
| Result interpretation, terminal gate, family verdict, reopen/promotion decision | `R3` / `frontier` | Required escalation before scientific interpretation or a verdict-changing write. |
| Unchanged-verdict route-branch archival or routine sync | `R2` / `balanced` | Dispatch when it is a standalone batch; verdict-changing sync remains `R3`. |

Supporting reads before an unresolved scientific verdict stay in the current
`R3` envelope. Once that verdict or design is durably recorded, a fresh
workspace/runner implementation is an `R2` Terra boundary; a later independent
preflight/launch/monitor batch is an `R1` Luna boundary; final interpretation
returns to an `R3` Sol task. This is the normal cost-saving route, not an
exception.

The route card must record the host identity mode, routing basis, planned task
boundaries, minimum roles, atomic batches, and whether each optional down-route
reduces projected whole-task tokens and credits after context reload. At an
eligible boundary, create the schema-valid request, run
`dispatch_agent_task.ps1` without `-Execute`, review `MODEL_DISPATCH_DRY_RUN_OK`,
then rerun with `-Execute`. The child owns exactly the request's `next_action`;
it must not continue into the next scientific or engineering class.

From a PowerShell session rooted at the route worktree, use the same reviewed
request for both calls:

```powershell
$request = "C:\path\to\dispatch-request.json"
$dispatcher = ".\experience_docx\tools\dispatch_agent_task.ps1"
& $dispatcher -RequestPath $request
& $dispatcher -RequestPath $request -Execute
```

Do not call the dispatcher from a cloud runner or inside a training process.
It selects the local agent task that will invoke the existing route setup,
`convir-ops`, monitoring, closeout, or sync tools. This keeps model-cost routing
orthogonal to GPU execution and experiment semantics.

If an eligible boundary is deliberately kept in a known qualified task because
the switch would not amortize, record `dispatch=not_amortized` and the bounded
remaining scope in the route card's agent-routing plan. Unknown hosts cannot
use `not_amortized`. If dispatch is required by role, `not_amortized` is never
an override.

## Whole-Task Token, Credit, And Time Budget

Optimize the remaining user-visible task, not the next tool call. Scientific
authority and written authorization are hard constraints. Among safe plans,
minimize total uncached input plus output/reasoning tokens first, then official
credit-equivalent cost and wall time. Track cached input separately; do not call
a high-cache plan token-free.

- For an unclassified or ambiguous large task, pay for at most one frontier
  planning child per continuous scope; reuse its durable routing plan until the
  scope changes.
- Start a fresh task at a terminal route decision or explicit major handoff.
- Compact only after the current closeout/evidence is durably committed; never
  compact away unsynced authorization or failure details.
- Read the router plus the minimum current-route evidence. Expand only at a
  written decision point.
- Use bounded MCP status/monitor results. Do not paste full stdout, raw tables,
  per-image results, or unrelated historical documents into the task.
- Keep routine updates to state, progress, primary metric, terminal marker, and
  exact next authorization.
- Reuse stable route cards, runner paths, and typed JSON instead of restating
  their contents in prompts.
- Do not use multiple agents for routine work. Temporary dual-model shadowing is
  allowed only for qualification or a recorded reliability audit.
- Do not use token caps that can interrupt a required launch audit, closeout, or
  sync halfway through. Budget by task boundary instead.

For every optional dispatch, compare the remaining-envelope alternatives:

```text
keep = incremental current-context work at the known host role
dispatch = request/handoff + child context reload + child work at target role
```

Dispatch only when the child is required for authority or the second plan is
expected to reduce total credits without a dominated increase in total
uncached tokens. When estimates are uncertain for one short adjacent action,
keep the known qualified context. When several same-class operations share the
same route commit, authorization, evidence set, and stop condition, group them
as one atomic batch. An atomic batch may include preflight, an already
authorized launch, and short monitoring only when all remain `R1` and the child
can stop before interpretation or engineering repair.

Amortize model switches. Keep adjacent `R0`/`R1` operations in one known
qualified `balanced` task when opening a fast task would reload more context
than the remaining work. Use one `fast` child for standalone status, repeated
or long monitoring, or a batch of identical bounded operations. Do not launch a
new child per poll. A task-scoped user pin avoids repeated identity checks but
does not justify keeping later independent work on a more expensive role.

Do not recompute the class downward after every read. First close the active
task envelope, then dispatch its independent continuation. This avoids both the
false economy of reloading context for one read and the opposite failure of
running every later bounded operation on the user's default frontier model.

Minimum handoff payload:

```text
rules_commit=<github/main SHA>
route_branch_commit=<SHA>
route_id=<id>
source_identity=<unknown or trusted source>
routing_basis=<dispatcher_classification or typed_handoff>
routing_basis_ref=<github:commit:path or none>
stage_state=<typed state>
decision=<typed decision>
authorizes=<exact next action or none>
cloud_status=<active/inactive and terminal marker>
next_action=<one bounded action>
```

Anything recoverable from those references should not be copied into the
handoff.

## Recommended Default

Use a task-scoped pin when the user explicitly fixes the current host model and
effort; otherwise record the host as unknown without asking. In both cases, the
host may route a clear class to its lowest qualified target. Route ambiguous
work to `frontier`/high, independent `R2` work to `balanced`/medium, and
amortized `R0`/exact `R1` batches to `fast` low/medium after the dated
qualifications. Return to `frontier`/high for scientific design or
interpretation.

An explicitly selected stronger interactive model is not inherited by
dispatcher children. Keep it only while its warm context is cheaper for the
remaining bounded envelope; otherwise select the cheapest qualified role at the
next durable boundary. This avoids both repeated cold starts and the much larger
cost of running the entire experiment lifecycle on the strongest model.

Official product references used for the dated mapping:

- <https://openai.com/index/gpt-5-6/>
- <https://developers.openai.com/api/docs/models>
- <https://developers.openai.com/api/docs/models/compare>
- <https://help.openai.com/en/articles/20001325-a-preview-of-gpt-56-sol-terra-and-luna>
- <https://learn.chatgpt.com/docs/models#choosing-sol-terra-and-luna>
- <https://learn.chatgpt.com/docs/models#pick-a-reasoning-effort>
- <https://learn.chatgpt.com/docs/config-file/config-reference>
