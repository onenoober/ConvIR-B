# Agent Instructions

## Hard Rules

- Local WSL is editing and syntax/compile-only. Never run tests, smoke,
  training, evaluation, inference, demos, or runtime commands locally.
- Runtime validation runs only on `convir-4090` unless the user explicitly
  overrides one command. Use
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- GitHub `main` owns current rules and terminal compact evidence. A named
  GitHub route branch owns runnable route code and intermediate compact
  evidence. Cloud owns runtime state and raw outputs. Local state is never
  experiment memory.
- Do not commit checkpoints, weights, datasets, images, arrays, archives, raw
  inference outputs, large per-image tables, selected-action tables, or raw
  feature tables by default.
- `github/codex/haze4k-official-arch-anchor` is immutable. New model-structure
  routes branch from it into fresh local and cloud workspaces.
- Do not treat chat history, an older route checkout, or a local branch label as
  current authority. Resolve the exact GitHub ref and commit.
- Keep one qualified current model for the complete warm task. A stronger model
  may perform every lower class directly. Do not create dispatcher children or
  split one experiment across models to save cost. Honor an explicit user
  model/effort pin.
- If the current model is below the minimum role, stop before the next write,
  launch, commit, push, or scientific decision and emit one
  `MODEL_SWITCH_REQUIRED` handoff. Do not claim or automate an in-task switch.
- A command, transport, path, validator, or implementation failure is
  engineering state. It never creates new scientific authority. Correct one
  command boundary once; allow one engineering repair cycle for one root cause;
  a repeated same-class failure produces one blocker and stops that operation.
- Keep one canonical source per rule. New rules follow the document layers in
  `experience_docx/README.md`; other files link instead of copying rule text.

## Universal Route Workflow

1. Classify the highest unresolved user-visible outcome under
   `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`; continue in the current qualified
   task.
2. Ground the route in GitHub evidence and current cloud state. Freeze route
   identity, data roles, locked-test policy, estimand, gates, and stop rules.
3. Complete the one-time bundle in
   `MODEL_EXPERIMENT_START_CHECKLIST.md`: one route card, one operations
   manifest, one runner, and only the route code/assets contract actually used.
4. Before each launch, follow `MODEL_RUN_OPERATIONS_PROTOCOL.md` and use the
   bounded `convir-ops` lifecycle when its tool matches the operation.
5. Interpret gates only with `EXPERIMENT_GOVERNANCE_PROTOCOL.md`. A typed
   closeout authorizes the next stage; engineering repair does not need a new
   scientific authorization when semantics remain frozen.
6. At a terminal decision or explicit major handoff, archive only compact text
   evidence under `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## Read Budget

Read the smallest useful set and expand only at a decision boundary.

| Task | Read |
| --- | --- |
| Model qualification | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` |
| Status/result/decision | GitHub `EXPERIMENT_INDEX.md`, then the relevant family summary, route card, evidence README/closeout, and current cloud state |
| New/changed route | `MODEL_EXPERIMENT_START_CHECKLIST.md` plus only relevant design/source policies |
| Launch/monitor/closeout | `MODEL_RUN_OPERATIONS_PROTOCOL.md`, `CONVIR_OPS_MCP.md`, and `COMMAND_RELIABILITY_QUICKSTART.md` |
| Evidence sync | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` and affected evidence files |
| Failed unfamiliar command boundary | Targeted section of `COMMAND_RELIABILITY_PROTOCOL.md` only after the single canonical correction is insufficient |

Never minimize away fact source, route identity, metric alignment, data-role
separation, resource preflight, stage authorization, locked-test policy, or
closeout provenance.

## Cloud And Sync Gates

- Inspect before launch; never overwrite an active session, output, or
  historical workspace.
- Every stage uses a tracked runner, explicit output id, status/heartbeat,
  stdout/stderr capture, and typed closeout.
- New routes use fresh workspaces. Exact continuation or exact unit-boundary
  resume must be explicitly supported by the frozen runner and manifest.
- Commit intermediate compact evidence to the route branch. Sync GitHub
  `main` only at terminal state or an explicit major handoff.
- Use a clean `github/main` worktree and explicit staged paths. Never
  force-push evidence.

## Command Reliability

- Prefer the six bounded `convir-ops` tools when their schema covers the
  task. For operations outside that boundary use
  `experience_docx/tools/convir_remote_script.sh <local-script>`.
- Use direct argument arrays for simple Git/path operations. Avoid nested
  PowerShell -> WSL -> shell quoting.
- Monitoring, sync, and audit commands print `*_OK` or write a status file.
- Partial output from a failed command is not scientific evidence. A timeout
  after the launch boundary is unknown state and requires inspection, never a
  blind retry.

When documentation conflicts with conversation, use current GitHub rules and
evidence plus current cloud runtime state. State uncertainty and cite the
authoritative path used.
