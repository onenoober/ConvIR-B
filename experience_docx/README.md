# Experiment Documentation Router

Date: 2026-07-16

Read only the source needed for the next decision.

| Need | Canonical source |
| --- | --- |
| current status/result | `EXPERIMENT_TERMINAL_INDEX.jsonl` -> exact contract/closeout/conclusion/results -> current cloud state when active |
| default experiment workflow | `SCIENCE_FASTPATH.md` |
| model qualification | `MODEL_QUALIFICATION_PROTOCOL.md` |
| scientific design/gates | `EXPERIMENT_GOVERNANCE_PROTOCOL.md` |
| fastest safe route build/launch | `ROUTE_READY_FASTPATH.md` |
| route amendment/assets/fixture/evidence helpers | `ROUTE_FLOW_TOOLS.md` |
| new or changed route | `MODEL_EXPERIMENT_START_CHECKLIST.md` and `EXPERIMENT_CARD_TEMPLATE.md` |
| cloud launch/monitor/closeout | `MODEL_RUN_OPERATIONS_PROTOCOL.md` and `CONVIR_OPS_MCP.md` |
| low-cost generic liveness telemetry | `GENERIC_RUN_MONITORING_PROTOCOL.md` |
| command transport | `COMMAND_RELIABILITY_PROTOCOL.md`; validation evidence in `COMMAND_TRANSPORT_V1_VALIDATION.md` |
| terminal scientific archive | `SCIENCE_FASTPATH.md` and `prepare_terminal_archive.py` |
| legacy/engineering evidence archive | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` and `validate_evidence_sync.py` |
| Haze4K architecture route | `OFFICIAL_ARCH_ANCHOR_POLICY.md` and `Haze4K_ARCH_FINETUNE_WORKFLOW.md` |

Authority is GitHub `main` for current rules and terminal evidence, the named
route branch for runnable/intermediate state, and `convir-4090` for raw runtime
state. Local files and chat are inputs, not project memory.

Historical reliability, workflow-evaluation, route, and incident documents are
provenance only. They do not define current defaults.
