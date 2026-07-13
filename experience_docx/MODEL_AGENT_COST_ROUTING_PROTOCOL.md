# Model-Agent Cost Routing Protocol

Date: 2026-07-13

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

Classify the current task before the first substantive tool call and again when
scope changes.

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

The active task cannot assume that a model switch happened. When the task class
requires a higher role than the active model:

1. stop before the next external write, launch, commit, push, or scientific
   decision;
2. emit `MODEL_SWITCH_REQUIRED` with the required stable role, current task
   class, exact durable handoff evidence, and the blocked next action;
3. resume in a new task or after an explicit model switch;
4. reread only the minimal durable handoff, not the prior chat transcript.

If the active model identity is unavailable, allow `R0` only. Do not claim an
automatic in-task switch without a verified product capability. A model may
down-route a later independent task, but it must not delegate high-risk meaning
to a lower role through a prompt or subagent.

## Token And Time Budget

Reduce tokens by shrinking context first and model price second.

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

Amortize model switches. Keep adjacent `R0`/`R1` operations such as preflight,
launch, short monitoring, and intermediate route-branch archival in one
qualified `balanced` task when opening a separate fast task would reload the
same context. Use a separate `fast` task for standalone status checks, repeated
or long monitoring, or batches of identical bounded operations. Choose the
lowest total context and model cost, not the lowest price for each individual
turn.

Minimum handoff payload:

```text
rules_commit=<github/main SHA>
route_branch_commit=<SHA>
route_id=<id>
stage_state=<typed state>
decision=<typed decision>
authorizes=<exact next action or none>
cloud_status=<active/inactive and terminal marker>
next_action=<one bounded action>
```

Anything recoverable from those references should not be copied into the
handoff.

## Recommended Default

Use `frontier` until the dated Terra/Luna qualification result exists. After
qualification, use `balanced`/medium for a new experiment task whose class is
not yet known. Down-route a fresh bounded task to `fast` only when classification,
qualification, exact authorization checks, and switch amortization all pass.
Escalate to `frontier` before scientific design or interpretation. This default
avoids paying frontier cost for routine operations without making a lower-cost
model the owner of an ambiguous or high-impact decision.

Official product references used for the dated mapping:

- <https://openai.com/index/gpt-5-6/>
- <https://developers.openai.com/api/docs/models>
- <https://developers.openai.com/api/docs/models/compare>
- <https://help.openai.com/en/articles/20001325-a-preview-of-gpt-56-sol-terra-and-luna>
- <https://learn.chatgpt.com/docs/config-file/config-reference>
