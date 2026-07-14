# Experiment Rules Router

Date: 2026-07-14

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
| L2 execution protocols | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`, `MODEL_EXPERIMENT_START_CHECKLIST.md`, `MODEL_RUN_OPERATIONS_PROTOCOL.md`, `CONVIR_OPS_MCP.md`, `COMMAND_RELIABILITY_QUICKSTART.md`, `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | Agent-model/task routing, one-time route setup, per-launch runtime work, bounded schema-v2 operations, command transport, and terminal evidence archival respectively. |
| L2 source policy | `OFFICIAL_ARCH_ANCHOR_POLICY.md`, `Haze4K_ARCH_FINETUNE_WORKFLOW.md` | Anchor/source rules plus Haze4K partial-load, initialization, and trainable-scope guidance. |
| L3 design aids | `ROUTE_DESIGN_FRAMEWORK.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md`, `CONVIR_B_EXECUTION_GUIDE.md` | Deeper design guidance, baseline contracts, and card templates. |
| L4 evidence indexes | `EXPERIMENT_INDEX.md`, `family_summaries/`, `experiment_cards/`, `experiment_logs/<route_id>/README.md` | Route state, decisions, evidence paths, and compact results. |
| L5 archives, cleanup maps, and rule audits | `COMMAND_RELIABILITY_PROTOCOL.md`, `CLOUD_PY310_ENVIRONMENT.md`, `BRANCH_ROUTE_INDEX.md`, `WORKFLOW_CHANGE_EVALUATION_20260712.md`, `SCIENTIFIC_FRAMEWORK_CHANGE_EVALUATION_20260713.md`, `MODEL_ROUTING_WORKFLOW_EVALUATION_20260713.md`, `MODEL_ROUTING_IDENTITY_CONTINUITY_EVALUATION_20260713.md`, `MODEL_ROUTING_UNKNOWN_HOST_TOTAL_COST_EVALUATION_20260714.md`, `UNIVERSAL_EXPERIMENT_OPS_V2_EVALUATION_20260714.md`, `R3_XHIGH_ROUTING_EVALUATION_20260715.md`, historical logs | Historical, deep troubleshooting, remote-branch cleanup, or workflow-change audit context; do not read by default. |

## Future Text Placement

All new rule or evidence text must stay in its proper layer:

- Put default execution rules and read-budget changes in `AGENTS.md`.
- Put task routing and document ownership changes in this router.
- Put agent-model roles, qualification, escalation, and token-budget rules in
  `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`.
- Put start/run/sync operational rules in the matching L2 protocol.
- Put the active bounded MCP schema and signed lifecycle only in
  `CONVIR_OPS_MCP.md`; historical tool schemas stay in L5 evidence.
- Put current command transport defaults in `COMMAND_RELIABILITY_QUICKSTART.md`;
  put only reusable failure history in `COMMAND_RELIABILITY_PROTOCOL.md`.
- Put route design rationale in L3 design aids, not in runtime protocols.
- Put experiment facts, decisions, and metrics in L4 evidence files, not in rule
  documents.
- Put historical environment or branch-cleanup context in L5 archives, not in
  current runtime rules.
- Put workflow-change evaluations in L5 audit notes; they may justify a rule
  change, but they are not execution protocols.

When a rule touches multiple files, update the canonical source and add short
pointers elsewhere. Do not duplicate the full rule body across layers.

## Execution Ownership

Use each operational document once for its own job:

| Moment | Canonical document | Output |
| --- | --- | --- |
| Before task execution or after a scope change | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` | task class, minimum qualified model role, and bounded handoff when escalation is required |
| Route creation or contract change | `MODEL_EXPERIMENT_START_CHECKLIST.md` | one route card and a selected decision design/profile |
| Every cloud stage launch and monitor | `MODEL_RUN_OPERATIONS_PROTOCOL.md` | durable runner, `status.txt`, and typed stage closeout |
| PowerShell/WSL/SSH boundary | `COMMAND_RELIABILITY_QUICKSTART.md` | a command with an explicit success/failure marker |
| Terminal decision or explicit major handoff | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | compact evidence on GitHub `main` |

Do not create separate current-state, runbook, manifest, workflow, and analysis
documents by default. The minimum route bundle is one route card, one durable
stage runner, `status.txt`, one typed `<stage>_closeout.json`, one evidence
README, and only the compact summaries required to support the decision.
Specialized contract files are justified only when they are independently
reusable or too large for the route card.

## Read Router

| Task | Read first | Then only if needed | Do not read by default |
| --- | --- | --- | --- |
| Choose an agent model, reduce token cost, compact, or hand off | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` | Relevant dated model-routing evaluation only for qualification or rule audit | Unrelated route history and full chat transcript |
| Current status, result, decision, or route memory | GitHub `main` or named branch copy of `EXPERIMENT_INDEX.md` | Relevant family summary, route card, evidence README/log dir | Local dirty checkout, chat history, unrelated route logs |
| New route, method, loss, selector, teacher, or audit | `MODEL_EXPERIMENT_START_CHECKLIST.md` | `ROUTE_DESIGN_FRAMEWORK.md`, `EXPERIMENT_CARD_TEMPLATE.md`, relevant family summary | Full governance package |
| Gate design, threshold, or scientific interpretation | `EXPERIMENT_GOVERNANCE_PROTOCOL.md` section `Gate Policy` | Relevant route card and metric contract | Unrelated historical gate files |
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
- Current workflow/governance rules come from GitHub `main`. Copies of these
  rule files inside older route branches, local worktrees, and cloud workspaces
  are historical snapshots and must not override current `main` rules.
- A directory name or local branch label such as `main`, `github-main`, or
  `main-sync` is not proof of authority. Resolve the `github/main` remote ref and
  record its exact commit before using current rules.
- `convir-4090` is the runtime/raw-output source.
- Local WSL is editing, syntax/static-check, sync staging, and local-safety
  inspection only.
- Attachments are task input, not durable project evidence until written into a
  route card, evidence README, closeout JSON, central index, or family summary.
- Historical cloud/environment files preserve provenance and should not override
  current `AGENTS.md` or runtime protocol rules.
- Existing route workspaces remain untouched for reproducibility. Start new
  routes in fresh workspaces; reuse an old workspace only for an explicitly
  authorized continuation or exact resume.

## Endpoint Responsibility Matrix

| Endpoint | Owns | Must not own |
| --- | --- | --- |
| Local WSL | rule editing, syntax/static checks, sync staging, local safety inspection | runtime validation, experiment status authority, raw-output truth |
| GitHub `main` | current process rules, terminal compact evidence, central status/index, family verdicts | raw outputs, checkpoints, route runtime workspaces, unpromoted experiment code |
| GitHub route branch | runnable route code, tracked runners, intermediate compact evidence for that route | global process rules, unrelated route memory, raw runtime artifacts |
| `convir-4090` | runtime execution, current active state, raw logs, checkpoints, images, arrays, large tables | current governance authority, final durable evidence archive |

The automation rule is simple: plan from GitHub `main`, run on cloud, keep raw
artifacts on cloud, and archive only compact terminal or major-handoff evidence
back to GitHub `main`. Local files may stage and inspect that process, but they
do not decide research status.

## Baseline Rule

ConvIR-B routes still require a trustworthy baseline contract before a model
change can make an improvement claim. Run evaluation and runtime validation only
on the authorized cloud runtime. Record only what is needed to reconstruct the
aggregate claim: checkpoint path/hash, code and explicit runtime, dataset/split
and verified sample count, preprocessing/metric settings, aggregate result, and
any reproduction gap. Add per-sample outputs, images, latency, memory, or visual
labels only when a written mechanism, failure-diagnosis, cost, or promotion gate
needs them.
