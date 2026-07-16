# Model-Agent Cost Routing Protocol

Date: 2026-07-16

Status: canonical L2 protocol for model qualification, task continuity, optional
dispatch, and model-cost control. It does not authorize experiment execution.

## Purpose And Authority

Classify the whole current task before experiment planning, implementation,
cloud operation, result interpretation, or evidence sync. Use the current
qualified model for the complete warm task envelope by default. Model routing
must never change the route commit, runner, data, checkpoint, metric contract,
gate, output root, authorization chain, or locked-test policy.

Current model-role mapping, based on the public GPT-5.6 family on 2026-07-13:

| Stable role | Current model | Default effort | Meaning |
| --- | --- | --- | --- |
| `frontier` | GPT-5.6 Sol | high | Highest-risk scientific and semantic decisions. |
| `balanced` | GPT-5.6 Terra | medium | Existing-contract engineering and reviewed writes. |
| `fast` | GPT-5.6 Luna | low or medium | Bounded, deterministic, mostly read-only operations. |

The stable role is canonical; the model name is a dated mapping. Update only
this table after a new qualification. Do not rewrite route evidence when a
model family changes.

Current qualification state:

| Role/model | Status | Maximum current class |
| --- | --- | --- |
| `frontier` / GPT-5.6 Sol | strongest-model baseline | `R3` |
| `balanced` / GPT-5.6 Terra | qualified 2026-07-13 | `R2` |
| `fast` / GPT-5.6 Luna | qualified 2026-07-13 | `R1` |

Qualification evidence:
`model_agent_qualification/20260713_gpt56/README.md`. Both candidate models
scored `91/91` critical fields with zero unauthorized actions under
`codex-cli 0.144.1`; the reviewer was `frontier` / GPT-5.6 Sol.

Change qualification only with a dated L5 result recording model id/version,
case manifest, exact critical-field score, unauthorized-action count, decision,
and frontier review. A workflow evaluation is not a qualification pass.

## Task Classes And Minimum Models

Classify the user-visible outcome and the highest-authority decision still open
inside the task. Supporting reads do not split an unresolved higher-class task.

| Class | Typical work | Minimum model | Allowed behavior |
| --- | --- | --- | --- |
| `R0_READ_ONLY` | status, bounded monitor, exact-path read, SHA/list/tree inspection, compact metric extraction | qualified `fast`, low | No writes and no scientific interpretation beyond typed-field reporting. |
| `R1_BOUNDED_EXECUTION` | MCP plan/start/finish of an already authorized runner, explicit compact evidence fetch, syntax/static validation | qualified `fast`, medium | Inputs remain verbatim from the current typed card/closeout; no redesign. |
| `R2_ENGINEERING_CONTROL` | route setup from a written design, transport repair, closeout integrity audit, unchanged-verdict evidence commit/sync | `balanced`, medium | Reviewed engineering changes only; scientific semantics remain frozen. |
| `R3_SCIENTIFIC_AUTHORITY` | bottleneck synthesis, route/gate/metric design, model or training change, result interpretation, ambiguous failure, family verdict, promotion, canary, locked test | `frontier`, high | Owns scientific meaning and difficult-to-reverse decisions. |

Hard assignments:

- Any change to data, split, checkpoint, optimizer, loss, model structure,
  metric, threshold, gate, or authorized stage is `R3`.
- Any terminal scientific `PASS`, `FAIL`, `INCONCLUSIVE`, promote/stop/reopen
  decision is `R3`, even when a script computed the metrics.
- Locked-test selection, launch authorization, interpretation, and post-result
  restrictions are always `R3`.
- Routine unchanged-verdict sync is `R2`; a verdict or reopen-condition change
  is `R3`.
- A command/transport failure is at least `R2`. Partial output from it is never
  scientific evidence.

Raise the class immediately when scope expands. Lower it only after the
higher-class decision is durably closed and a bounded next action is written.
A stronger qualified model or higher effort may perform every lower class
directly. Keeping a warm stronger-model context requires no exception or
dispatcher justification.

## Identity, Effort, And Switching

Accept current product metadata, a verified dispatcher receipt, or an explicit
user model/effort pin for the current interactive task. A user pin survives
ordinary turns and context compaction; invalidate it on a new/forked task, an
explicit selector change, conflicting product metadata, loss of continuity, or
user withdrawal. Shell environment variables are not identity evidence.

Reasoning effort is a minimum: `low < medium < high < xhigh`. A higher effort
satisfies a lower minimum but grants no extra experiment authority. Use this
marker before substantive work and when scope changes:

```text
MODEL_ROUTE class=<class> role=<stable-role> effort=<active-effort>
```

`R3` uses `high` by default. Select Sol `xhigh` only when deeper reasoning can
materially change a high-impact decision and at least one condition holds:

- locked-test, canary, promotion, stop/reopen, or another difficult-to-reverse
  decision is owned by this task;
- evidence or failure classification is conflicting enough to change the
  authorized continuation;
- interacting changes to structure, data/split, metric, estimand, gate, or
  adaptive branches are designed together;
- cross-route synthesis must reconcile several durable evidence sources.

When a known active model is below the class minimum, stop before the next
external write, launch, commit, push, or scientific decision and emit one
`MODEL_SWITCH_REQUIRED` handoff containing the required role, class, durable
evidence references, and one blocked next action. Resume only after an explicit
model switch or in a new qualified task. Do not claim an in-task switch without
a verified product capability.

An unknown host may perform `R0` reporting. For `R1`-`R3`, establish a known
qualified model or stop with the same single switch handoff. An explicitly
enabled dispatcher may satisfy that handoff, but it is never mandatory.

## Non-Negotiable Experiment Invariants

No model choice or cost optimization may weaken these facts:

- GitHub `main` or the named GitHub route branch is experiment memory;
- local WSL remains editing and syntax/compile-only;
- runtime remains `convir-4090` with the explicit cloud Python;
- a new route uses a fresh workspace and its authorized source commit;
- every launch has current dynamic preflight, a tracked runner, fresh session
  and output directory, status/log capture, and typed closeout;
- the previous closeout explicitly authorizes the exact launched stage;
- formal gates use the written metric contract and canonical Gate Policy;
- locked test remains blocked without explicit prior authorization;
- compact evidence selection, staging, push, and remote verification remain
  unchanged.

If a proposed switch or smaller context would skip one invariant, reject it.

## Qualification Gate

`fast` may perform `R0`. It may perform `R1` only while its dated qualification
is current and the tracked runner or MCP operation machine-checks the exact
`route_id`, `state`, `decision`, and `authorizes` tuple from a GitHub typed JSON
handoff. A state-only or prose-only check is insufficient. Ambiguity requires
`frontier`; an exact engineering audit may use qualified `balanced`.

Qualification must have exact agreement on fact source; route id, branch,
commit, stage, runner, and output root; authorization tuple and failure class;
locked-test/canary policy; evidence allowlist; and stop/continue/escalate. The
acceptance line is `100%` on safety-critical fields with zero unauthorized
writes or continuations. Repeat after a model-family/version change or a
safety-critical execution error.

## Optional External Dispatcher

`experience_docx/tools/dispatch_agent_task.ps1` is an optional cost tool, not
the experiment control plane. It is default-disabled. Use it only when the user
explicitly enables dispatcher execution for the current task and one of these
cases has a concrete net benefit:

- long or repeated read-only monitoring with one stable authorization;
- an amortized batch of bounded same-class operations;
- an explicit major handoff or a required model switch.

Do not dispatch an adjacent short operation, routine supporting read, transport
repair, or warm continuation. Never dispatch merely because a cheaper role
exists. Compare the complete remaining alternatives:

```text
keep = incremental work in the current qualified warm context
dispatch = request preparation + child context reload + child work + review
```

Dispatch only when it is required for qualification or is expected to reduce
total official credits without increasing uncached tokens enough to dominate
that saving. Batch operations only when class, route commit, authorization
tuple, evidence set, and stop condition are identical.

The launcher validates the schema-v2 request, current rules commit, role floor,
route HEAD, typed `R1` authorization, execution scope, and transport contract.
Dry-run makes zero model calls. Actual execution additionally requires both
`-Execute` and `-EnableOptionalDispatch`; neither flag authorizes an experiment
stage. Raw child events stay outside the repository.

The launcher provides the verified rule, repository, handoff, path, and
transport constraints. A child must:

- acknowledge its route marker and handoff before any tool call;
- execute exactly one bounded next action using supplied paths;
- not call the dispatcher, create another model task, or reload the routing
  skill/protocol;
- use the named structured MCP operation for covered cloud work and avoid
  free-form SSH, PowerShell, WSL, Git, or shell-form selection;
- report one typed success or failure and stop at the boundary.

If evidence requires a stronger role, the child emits one
`MODEL_SWITCH_REQUIRED` result and stops. It must not produce or execute another
dispatcher request. The current interactive task owns any recovery.

### Dispatch Circuit Breaker

The authorization identity is the exact tuple:

```text
route_id + stage_state + decision + authorizes
```

After an actual child starts but fails dispatcher acceptance, the launcher
writes a persistent breaker record keyed by the SHA-256 of that tuple. Any
later `-Execute` for the same tuple fails before a model call. Dispatcher
failure must never cause automatic reclassification, authorization generation,
or another child.

Recovery is deliberately manual: diagnose in the current qualified task,
record the resolved cause, verify that retrying the same authorization is still
scientifically valid, then remove the named breaker file before a new explicit
dispatch. The launcher has no retry or clear-breaker switch.

### Bounded Child Contract

Every `R1` request uses `routing_basis=typed_handoff` and binds
`routing_basis_ref` to the exact GitHub JSON whose
`route_id/state/decision/authorizes` fields are copied into the request. A
caller-provided `verified=true` is insufficient without an exact evidence
match. The active route lifecycle remains:

```text
convir_route_plan_manifest -> convir_route_start_authorized -> convir_route_finish
```

The dispatcher does not run in a cloud runner or training process. It may
launch only one ephemeral Codex child. A child cannot cross into a different
task class, scientific interpretation, repair, or evidence verdict.

Use one reviewed request for dry-run and optional execution:

```powershell
$request = "C:\path\to\dispatch-request.json"
$dispatcher = ".\experience_docx\tools\dispatch_agent_task.ps1"
& $dispatcher -RequestPath $request
& $dispatcher -RequestPath $request -Execute -EnableOptionalDispatch
```

## Bounded Engineering Recovery

Command-boundary recovery is owned by
`COMMAND_RELIABILITY_QUICKSTART.md`. In summary, one deterministic correction
is allowed before launch; a second failure at the same boundary stops with one
blocker. Never use this recursive flow:

```text
failure -> new authorization -> dispatcher child -> failure
```

A command failure is engineering state, not evidence and not permission to
change a route, command contract, threshold, or scientific decision.

## Whole-Task Budget

Optimize the remaining user-visible task, not each tool call. In order:

1. preserve scientific authority and written authorization;
2. minimize total uncached input plus output/reasoning tokens;
3. minimize official credit-equivalent cost;
4. minimize wall time without weakening the first three constraints.

Practical defaults:

- keep one qualified current task across adjacent classes it is allowed to own;
- read the router plus the minimum current-route evidence and expand only at a
  written decision point;
- use bounded MCP results rather than full stdout, raw tables, images, or chat
  transcripts;
- reuse route cards, runner paths, typed JSON, and compact closeouts by durable
  reference;
- do not create multiple agents for routine work or one child per poll;
- compact only after authorization and failure details are durably recorded;
- do not use token caps that can interrupt launch audit, closeout, or sync.

Minimum durable switch or optional-dispatch payload:

```text
rules_commit=<github/main SHA>
route_branch_commit=<SHA>
route_id=<id>
routing_basis=<dispatcher_classification or typed_handoff>
routing_basis_ref=<github:commit:path or none>
stage_state=<typed state>
decision=<typed decision>
authorizes=<exact next action or none>
cloud_status=<active/inactive and terminal marker>
next_action=<one bounded action>
```

Anything recoverable from these references stays out of the handoff.

## Recommended Default

Run the whole warm task on the current qualified model. Honor an explicit user
pin. A frontier model may directly perform `R0`-`R3`; a balanced model may
perform `R0`-`R2`; a fast model may perform qualified `R0`-`R1`. Switch only
when the active role is below the class floor. Keep the dispatcher off unless
the user opts in and a long bounded batch or major handoff clearly amortizes
the cold start. Return to `frontier` before scientific design, interpretation,
promotion, canary, or locked-test work.

Official product references used for the dated mapping:

- <https://openai.com/index/gpt-5-6/>
- <https://developers.openai.com/api/docs/models>
- <https://developers.openai.com/api/docs/models/compare>
- <https://help.openai.com/en/articles/20001325-a-preview-of-gpt-56-sol-terra-and-luna>
- <https://learn.chatgpt.com/docs/models#choosing-sol-terra-and-luna>
- <https://learn.chatgpt.com/docs/models#pick-a-reasoning-effort>
- <https://learn.chatgpt.com/docs/config-file/config-reference>
