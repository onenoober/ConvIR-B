# Experiment Rules Router

Date: 2026-06-10

Status: lightweight router for model experiment rules.

## Purpose

Use this file only as a router. It tells an agent which rule file to read for a
specific task and which files to avoid by default. `AGENTS.md` remains the
highest-level instruction source; this package provides the repository-specific
details behind those rules.

The default thinking and execution flow is:

```text
fact source -> route identity -> resource preflight -> metric contract ->
reliable transport -> stage gates -> closeout/archive
```

Do not load every governance file just because a task mentions an experiment.
Read the smallest layer that can answer the current question.

For the adaptive context rule, use `AGENTS.md` section
`Adaptive Context Expansion`: start small, then expand at gate/decision points;
do not reduce any required gate.

## Document Layers

| Layer | Files | Role |
| --- | --- | --- |
| L0 default rules | `AGENTS.md` | Hard execution rules, fact-source authority, Universal Route Workflow, and read budget. |
| L1 router | `experience_docx/README.md` | This file; choose the next minimal rule/evidence file. |
| L2 execution protocols | `MODEL_EXPERIMENT_START_CHECKLIST.md`, `MODEL_RUN_OPERATIONS_PROTOCOL.md`, `COMMAND_RELIABILITY_QUICKSTART.md`, `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | Required checklists for starting, running, transporting, monitoring, and syncing work. |
| L2 source policy | `OFFICIAL_ARCH_ANCHOR_POLICY.md`, `Haze4K_ARCH_FINETUNE_WORKFLOW.md` | Clean-route and architecture/fine-tune rules for Haze4K model-structure work. |
| L3 design aids | `ROUTE_DESIGN_FRAMEWORK.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md`, `CONVIR_B_EXECUTION_GUIDE.md` | Deeper design guidance, baseline contracts, and card templates. |
| L4 evidence indexes | `EXPERIMENT_INDEX.md`, `family_summaries/`, `experiment_cards/`, `experiment_logs/<route_id>/README.md` | Route state, decisions, evidence paths, and compact results. |
| L5 archives and cleanup maps | `COMMAND_RELIABILITY_PROTOCOL.md`, `CLOUD_PY310_ENVIRONMENT.md`, `BRANCH_ROUTE_INDEX.md`, historical logs | Historical, deep troubleshooting, or remote-branch cleanup context; do not read by default. |

## Future Text Placement

All new rule or evidence text must stay in its proper layer:

- Put default execution rules and read-budget changes in `AGENTS.md`.
- Put task routing and document ownership changes in this router.
- Put start/run/sync operational rules in the matching L2 protocol.
- Put current command transport defaults in `COMMAND_RELIABILITY_QUICKSTART.md`;
  put only reusable failure history in `COMMAND_RELIABILITY_PROTOCOL.md`.
- Put route design rationale in L3 design aids, not in runtime protocols.
- Put experiment facts, decisions, and metrics in L4 evidence files, not in rule
  documents.
- Put historical environment or branch-cleanup context in L5 archives, not in
  current runtime rules.

When a rule touches multiple files, update the canonical source and add short
pointers elsewhere. Do not duplicate the full rule body across layers.

## Read Router

| Task | Read first | Then only if needed | Do not read by default |
| --- | --- | --- | --- |
| Current status, result, decision, or route memory | GitHub `main` or named branch copy of `EXPERIMENT_INDEX.md` | Relevant family summary, route card, evidence README/log dir | Local dirty checkout, chat history, unrelated route logs |
| New route, method, loss, selector, teacher, or audit | `MODEL_EXPERIMENT_START_CHECKLIST.md` sections `0` and `0B` | `ROUTE_DESIGN_FRAMEWORK.md`, `EXPERIMENT_CARD_TEMPLATE.md`, relevant family summary | Full governance package |
| New Haze4K model-structure route | `OFFICIAL_ARCH_ANCHOR_POLICY.md` and `Haze4K_ARCH_FINETUNE_WORKFLOW.md` | `CONVIR_B_EXECUTION_GUIDE.md` baseline contract, route card template | Old failed route branches unless explicitly continuing |
| Cloud launch, monitoring, eval, inference, or post-run audit | `MODEL_RUN_OPERATIONS_PROTOCOL.md` and `COMMAND_RELIABILITY_QUICKSTART.md` | `COMMAND_RELIABILITY_PROTOCOL.md` only for failed/unfamiliar command-boundary cases | Historical command failures |
| Evidence sync to GitHub `main` | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | Affected index/card/family/evidence README | Route code unless promotion is explicitly approved |
| Remote branch cleanup or retained-branch audit | `BRANCH_ROUTE_INDEX.md` | GitHub remote branch list and affected evidence index entries | Experiment route logs unrelated to cleanup |
| Command quoting, CRLF, PATH, stdin, or SSH failure | `COMMAND_RELIABILITY_QUICKSTART.md` | Targeted section in `COMMAND_RELIABILITY_PROTOCOL.md` | Full protocol scan |
| General experiment design question | `ROUTE_DESIGN_FRAMEWORK.md` | `EXPERIMENT_GOVERNANCE_PROTOCOL.md` for deeper rationale | Runtime protocols |
| Environment migration or old cloud provenance | `CLOUD_PY310_ENVIRONMENT.md` | Matching historical evidence logs | Current runtime decisions |

## Authority Boundaries

- GitHub `main` and named GitHub route branches are the durable compact evidence
  sources for status, decisions, and route memory.
- `convir-4090` is the runtime/raw-output source.
- Local WSL is editing, syntax/static-check, sync staging, and local-safety
  inspection only.
- Attachments are task input, not durable project evidence until written into a
  route card, evidence README, closeout JSON, central index, or family summary.
- Historical cloud/environment files preserve provenance and should not override
  current `AGENTS.md` or runtime protocol rules.

## Baseline Rule

ConvIR-B routes still require a trustworthy baseline contract before a model
change can make an improvement claim. Run evaluation and runtime validation only
on the authorized cloud runtime, record checkpoint path/hash and matched metric
settings, and explain any reproduction gap before promotion.
