# Model Routing Identity And Continuity Evaluation

Date: 2026-07-13

Status: `PASS_RULE_CLARIFICATION_NO_DISPATCHER_CODE_CHANGE`.

This L5 workflow-change audit supports
`MODEL_AGENT_COST_ROUTING_PROTOCOL.md`. It is not an execution protocol, model
qualification result, or experiment decision.

## Observed Failure

An interactive Codex App task was visibly configured as GPT-5.6 Sol with
`xhigh` effort. A tool subprocess exposed no `CODEX_MODEL`-style environment
variable. The missing variable was incorrectly treated as proof that the active
model identity was unavailable, so an `R3` scientific review stopped after
read-only evidence collection.

No model switch had occurred. The Sol task was overqualified for its supporting
reads, but the reads were incorrectly separated from the unresolved `R3` task
envelope. The equality wording around `high` also failed to state that `xhigh`
satisfies a `high` minimum.

## Root Causes

1. The protocol did not name authoritative identity sources for the Codex App,
   CLI, and dispatcher surfaces.
2. It did not state that missing shell environment variables are non-evidence.
3. It classified operations without an explicit continuity rule for supporting
   reads inside a higher-class user-visible outcome.
4. It did not distinguish an in-task model change from the dispatcher's actual
   behavior: launching a new ephemeral task with explicit model and effort.

## Rule Change

The canonical L2 protocol now:

- accepts dispatcher receipts, product metadata, current CLI `/status`, or a
  current-turn interactive Codex App selector confirmation in that order;
- treats `xhigh` as satisfying a `high` minimum;
- preserves the highest unresolved class across supporting operations;
- lowers a class only at a durable handoff with an independent bounded action;
- requires eligible standalone or batched lower-class work to down-route by
  default after that boundary;
- keeps `dispatch=not_amortized` only for a concrete context-cost reason; and
- states that dispatcher children explicitly override the user's default model.

The current-turn user confirmation is task configuration only. It cannot be
used as experiment evidence or propagated to unattended children.

## Cost Preservation

The dispatcher implementation and schema are unchanged. Their dated validation
already demonstrates explicit child selection and remains applicable:

| New task envelope | Qualified target | Dispatch behavior |
| --- | --- | --- |
| standalone/repeated `R0` or exact authorized `R1` | GPT-5.6 Luna | explicit child `--model`; low/medium effort |
| independent `R2` engineering handoff | GPT-5.6 Terra | explicit child `--model`; medium effort |
| `R3` design, interpretation, or escalation | GPT-5.6 Sol | explicit child `--model`; high effort |

A typical experiment therefore keeps evidence reads with the unresolved Sol
decision, hands written implementation to Terra, hands repeated monitoring to
Luna, and returns terminal interpretation to Sol. The user's default model is
not inherited across these dispatcher boundaries.

## Validation Boundary

This change modifies documentation and task classification only. It does not
change the dispatcher, request schema, qualification tables, model mapping,
runner, cloud command, experiment metric, or authorization tuple. The existing
dispatcher dry-run and end-to-end results remain the runtime evidence. Local
work is limited to text/static checks under repository policy.

Decision:
`IDENTITY_SOURCE_AND_TASK_ENVELOPE_CLARIFIED_KEEP_VALIDATED_COST_ROUTING`.
