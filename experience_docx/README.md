# Experiment Rules Router

Date: 2026-07-16

Status: lightweight router; do not load all governance files by default.

The default flow is:

```text
GitHub facts -> route identity/estimand -> minimal bundle -> cloud preflight ->
authorized stage -> typed closeout -> terminal interpretation/archive
```

## Layers

| Layer | Canonical files | Purpose |
| --- | --- | --- |
| L0 | `AGENTS.md` | non-negotiable endpoint, model, recovery and archive rules |
| L1 | this file | choose the smallest next source |
| L2 | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`, `MODEL_EXPERIMENT_START_CHECKLIST.md`, `MODEL_RUN_OPERATIONS_PROTOCOL.md`, `CONVIR_OPS_MCP.md`, `COMMAND_RELIABILITY_QUICKSTART.md`, `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` | qualification, setup, runtime, transport and archive |
| L2 source | `OFFICIAL_ARCH_ANCHOR_POLICY.md`, `Haze4K_ARCH_FINETUNE_WORKFLOW.md` | source/load/init/freeze rules |
| L3 | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md`, `EXPERIMENT_CARD_TEMPLATE.md` | scientific design and generic template |
| L4 | `EXPERIMENT_INDEX.md`, family summaries, route cards, evidence README/closeouts | current route memory |
| L5 | reliability/history/incident documents | read only for a matching audit or failure; no current authority |

## Minimal Reads

| Task | Start with | Expand only when |
| --- | --- | --- |
| model qualification | model qualification protocol | active model is below the role floor |
| status/result | index -> route card -> evidence README/closeout | decision needs family/cloud context |
| new route | start checklist + relevant source/design section | a written gate or design decision requires it |
| cloud stage | run protocol + MCP + quickstart | command boundary is uncovered or fails once |
| archive | sync protocol + affected files | terminal or explicit major handoff |

Historical route checkouts and deleted automation commits preserve provenance
only. They must not restore old dispatcher, stage, path, monitoring, or sync
defaults.

## Endpoint Ownership

| Endpoint | Owns |
| --- | --- |
| Local WSL | editing, syntax/compile checks, safe sync staging |
| GitHub `main` | current rules, terminal compact evidence, central index/family verdict |
| GitHub route branch | runnable route code and intermediate compact evidence |
| `convir-4090` | runtime state, raw logs, checkpoints, arrays, images and large tables |

Attachments and chat are inputs, not project memory, until represented by one
of the durable L4 sources.
