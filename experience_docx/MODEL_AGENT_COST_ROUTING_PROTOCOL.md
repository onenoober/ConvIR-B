# Model Qualification Protocol

Date: 2026-07-16

Status: canonical L2 protocol for task authority and current-model
qualification. It does not authorize an experiment stage.

## Core Rule

Classify the highest-authority unresolved user-visible outcome, then keep the
complete warm task in the current qualified model. Supporting reads,
engineering, launch, monitoring, closeout, and archive do not require separate
model tasks when the current model is already qualified.

| Class | Work | Minimum role |
| --- | --- | --- |
| `R0_READ_ONLY` | exact status, bounded monitoring, path/SHA inspection | `fast` |
| `R1_BOUNDED_EXECUTION` | exact authorized MCP operation, syntax/static validation | `fast` |
| `R2_ENGINEERING_CONTROL` | frozen-contract implementation, transport repair, unchanged-verdict sync | `balanced` |
| `R3_SCIENTIFIC_AUTHORITY` | route/gate/metric design, training/model change, interpretation, promotion, canary, locked test | `frontier` |

Current mapping: `frontier=GPT-5.6 Sol`, `balanced=GPT-5.6 Terra`, and
`fast=GPT-5.6 Luna`. The stable role is canonical; dated names may change
after qualification.

Hard assignments:

- data, split, checkpoint, optimizer, loss, model, metric, threshold, gate, or
  authorized-stage changes are `R3`;
- terminal scientific verdicts, promotion, canary, and locked test are `R3`;
- semantic-preserving engineering repair is `R2`;
- command/transport failure is engineering state and partial output is never a
  scientific result.

State before substantive work or when scope rises:

```text
MODEL_ROUTE class=<class> role=<stable-role> effort=<level>
```

Higher roles and efforts satisfy lower minima. A user model/effort pin survives
ordinary turns and compaction in the same task.

## Qualification Failure

If the active model is below the class minimum or unknown for `R1-R3`, stop
before the next external write, launch, commit, push, or scientific decision.
Emit exactly one handoff:

```text
MODEL_SWITCH_REQUIRED
required_role=<role>
class=<class>
evidence=<durable GitHub/cloud references>
blocked_next_action=<one action>
```

Resume only after the user or product explicitly provides a qualified model.
Do not create a child task, call a dispatcher, or claim an automatic switch.

## Removed Dispatcher

The external experiment dispatcher is retired. It is not an optional path and
must not be restored through a Skill, route card, run protocol, recovery rule,
or model-visible tool. Historical dispatcher evidence is archive-only and has
no current authority.

## Finite Recovery And Cost Control

Optimize the whole remaining user-visible task:

1. preserve scientific meaning and safety;
2. minimize uncached context by durable reference;
3. avoid repeated reads and repeated authorization;
4. minimize runtime without weakening the first three.

Use these hard budgets:

- model-task/child creation: zero;
- one deterministic correction per command-boundary class;
- one engineering repair cycle per root cause before a single blocker;
- one route-card validation before freeze and again only after a relevant
  contract change;
- one terminal interpretation and one terminal/major-handoff archive.

Do not create authorization, closeout, or Git commits for validator predicates,
path corrections, read timeouts, negative fixtures, positive controls, or
transport retries. Keep the same scientific authorization when semantics are
unchanged.

## Context Contract

Start with this protocol plus the minimum current-route evidence. Expand only
at route design, launch, terminal interpretation, locked test, or archive.
Never carry raw event streams, large tables, full logs, or the full chat into a
new context.
