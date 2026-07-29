# Experiment Documentation Router

Date: 2026-07-26

Read only the source needed for the next decision.

| Need | Canonical source |
| --- | --- |
| current status/result | compact `convir_git_status.authoritative_snapshot` -> referenced contract/closeout/conclusion/results; read the full terminal index only for conflict repair |
| cross-route evidence discovery | `CONVIR_EVIDENCE_REVIEW.md`; catalog summary -> filtered query -> explicit reads of only selected evidence |
| default experiment workflow | `SCIENCE_FASTPATH.md` |
| model qualification | `MODEL_QUALIFICATION_PROTOCOL.md` |
| scientific design/gates | `EXPERIMENT_GOVERNANCE_PROTOCOL.md` |
| fastest safe route build/launch | `ROUTE_READY_FASTPATH.md` |
| route amendment/assets/fixture/evidence helpers | `ROUTE_FLOW_TOOLS.md` |
| new or changed route | `ROUTE_READY_FASTPATH.md`; research-program contract plus one schema-2 experiment spec compiled to route schema 6 |
| compact AI rule routing | `AI_POLICY_SNAPSHOT.json` first, then every full authority file routed by its change class |
| exact engineering qualification reuse | `capability_registry.jsonl` and `tools/capability_registry.py`; engineering-only, six-field exact identity |
| complete-unit recovery | `ROUTE_READY_FASTPATH.md`; hash-bound ledger asset plus verified unit outputs in a fresh run |
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
