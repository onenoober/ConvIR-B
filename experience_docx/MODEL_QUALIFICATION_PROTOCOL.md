# Model Qualification Protocol

Date: 2026-07-16

This protocol checks capability only. It does not route work, create tasks,
switch models, or authorize an experiment stage.

| Class | Work | Minimum role |
| --- | --- | --- |
| `R0` | status/path/SHA reads and bounded observation | fast |
| `R1` | exact authorized operation and static validation | fast |
| `R2` | frozen-contract implementation and transport repair | balanced |
| `R3` | scientific design, model/data/gate changes, interpretation, promotion, sealed use | frontier |

Current role mapping is `frontier=GPT-5.6 Sol`, `balanced=GPT-5.6 Terra`, and
`fast=GPT-5.6 Luna`. Higher roles satisfy lower floors. A user model/effort pin
wins.

Check once when qualification is requested or uncertain. If qualified, continue
the entire warm workflow in the same task. If not, stop before the next write,
launch, commit, push, or scientific decision and report the required role,
durable evidence, and one blocked action. Never dispatch, create a child task,
or claim an automatic/in-task switch.

Command and engineering failures never trigger model escalation or create
scientific authority.
