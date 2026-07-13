# Agent Instructions

## Hard Rules

- Local WSL = editing and syntax/compile-only checks. Do not run tests, smoke
  tests, training, evaluation, inference, demos, or runtime commands locally.
- Runtime validation runs only on `convir-4090` unless the user explicitly
  overrides a specific command. If unavailable, report it; do not fall back
  locally.
- Use explicit cloud Python paths, especially
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- GitHub `main` = durable compact evidence archive and current experiment
  status source. Cloud = runtime/raw-output source. Local = editing/sync
  staging only.
- Do not commit checkpoints, weights, datasets, images, arrays, archives, raw
  inference outputs, large per-image tables, selected-action tables, or raw
  feature tables by default.
- `github/codex/haze4k-official-arch-anchor` is immutable. New model-structure
  routes must branch from it.
- For experiment status, result, decision, and route-memory questions, read
  GitHub `main` or the named GitHub branch directly. Do not use the local
  working tree, local `experience_docx/`, or local git state as project memory;
  they may be dirty, stale, or on the wrong branch. Local reads are allowed only
  for editing, syntax/compile-only checks, sync staging, and verifying whether a
  local operation is safe.
- Do not treat chat history as authoritative evidence.
- Before substantive experiment work, classify the task and use the cheapest
  qualified model allowed by
  `experience_docx/MODEL_AGENT_COST_ROUTING_PROTOCOL.md`. Model routing must not
  weaken any experiment gate or change route semantics. If the active model is
  below the required role, stop before external writes or scientific decisions
  and emit a durable handoff for an explicit switch.
- At an eligible new experiment task boundary, explicit major handoff, required
  escalation, or amortized batch of bounded operations, use the deterministic
  dispatcher from that protocol as the default task launcher. Run its dry-run
  first and add `-Execute` only after the handoff and scientific authorization
  are complete. Keep adjacent short operations in the current qualified task
  when the protocol says a switch would not amortize.
- New rule text must follow the repository document layers in
  `experience_docx/README.md`. Keep one canonical source per rule; other files
  should link to that source instead of copying the rule body. Do not mix
  runtime rules, design guidance, historical troubleshooting, branch cleanup, or
  experiment evidence in the same document section.
- Current process rules come from GitHub `main`. Rule files inside an older
  local/cloud route checkout are historical snapshots, not current authority.
  Preserve their route evidence, but do not let their old stage, sync, path, or
  monitoring defaults govern new work. When a workflow step names a rule path,
  read that path from a freshly fetched `github/main`; read a local copy only
  when the task is editing that rule.
- A new route uses a fresh local/cloud workspace. Reuse an existing workspace
  only for an explicitly named continuation or exact resume after checking its
  dirty state; never repurpose a historical route directory for a new route.

## Universal Route Workflow

Use this order for every route; the matching L2 document owns the detailed
checks. Do not repeat those checklists in route cards or other rule files.

1. Classify the current agent task and execute any required or amortized model
   dispatch under `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`.
2. Ground the task in GitHub evidence and current cloud state.
3. Classify the experiment route and record forbidden continuations and
   locked-test policy.
4. Complete the one-time route setup in
   `experience_docx/MODEL_EXPERIMENT_START_CHECKLIST.md`.
5. Before each cloud launch, complete the dynamic preflight and run only the
   authorized stage from `experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md`.
6. Use `experience_docx/COMMAND_RELIABILITY_QUICKSTART.md` for transport.
7. Interpret formal gates only with the canonical Gate Policy in
   `experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md`.
8. At a terminal route decision or explicit major handoff, archive compact text
   evidence with `experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## Read Budget

Read the smallest useful set; do not open all governance docs by default. Use
`rg` and targeted excerpts. Stop reading once the task is grounded.

### Adaptive Context Expansion

Use a small starting context, then expand only at decision points. The
non-negotiable gates remain fact source, route identity, resource preflight,
metric contract, stage authorization, and closeout/archive.

- Start with the router plus the minimum current-route evidence needed for the
  question.
- Expand to the relevant L2 protocol before new route start, cloud launch,
  runtime monitoring, evidence sync, or command execution.
- Expand to route card, family summary, evidence README/log status, and cloud
  state before interpreting results or deciding pass/fail/pause/promote.
- Expand to historical archives only for command-boundary failures, old cloud
  provenance, or branch-cleanup tasks.
- Never use context minimization to skip locked-test policy, forbidden-flow
  checks, metric alignment, resource preflight, or written stage gates.

| Task | Read |
| --- | --- |
| Model choice, token/cost control, or task handoff | `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` |
| Experiment status/result/decision | GitHub `main`/named GitHub branch copy of `EXPERIMENT_INDEX.md`, then only the relevant family summary, route card, and evidence README/log dir |
| Cloud command, monitoring, sync, PowerShell/WSL/SSH | `COMMAND_RELIABILITY_QUICKSTART.md`; read `COMMAND_RELIABILITY_PROTOCOL.md` only for failed or unfamiliar command-boundary cases |
| Training, smoke, eval, inference, post-run audit | `MODEL_RUN_OPERATIONS_PROTOCOL.md` |
| Evidence sync to GitHub | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`, affected index/card/family/README |
| New Haze4K architecture/fine-tune route | `Haze4K_ARCH_FINETUNE_WORKFLOW.md`, partial-load/init/freeze rules |
| New experiment family/governance | Relevant sections only from `README.md`, governance/checklist/design/template docs |

## Sync Gates

- Sync only valuable compact text evidence, not raw artifacts or broad outputs.
- Commit intermediate compact evidence to the route branch. Sync GitHub `main`
  only at a terminal route state or an explicitly recorded major handoff.
- Use a clean `github/main` worktree, stage explicit paths, and follow
  `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` for path selection, checks, push, and
  verification.

## Cloud Gates

- Before launch, satisfy the resource-preflight and metric-contract gates in
  the Universal Route Workflow and `MODEL_RUN_OPERATIONS_PROTOCOL.md`.
- Record the GitHub `main` rules commit used for planning. Cloud route checkouts
  provide code and runtime state, not governance authority.
- Do not overwrite active sessions, output dirs, or model names; inspect first.
- Every cloud run needs a durable command script, heartbeat/status, stdout/stderr
  capture, and compact evidence closeout.
- Distinguish infra/preflight/training/eval/scientific-gate failures explicitly.

## Command Reliability

- Follow the MCP-first transport selection rule in
  `experience_docx/COMMAND_RELIABILITY_QUICKSTART.md`. When its bounded
  `convir-ops` operation matches the user task and the MCP is registered, select
  that tool directly; use the script wrapper only for operations outside that
  scope or when the MCP is unavailable.
- For PowerShell -> WSL -> SSH, prefer
  `experience_docx/tools/convir_remote_script.sh <local-script>` over nested
  quoting.
- Monitoring/sync/audit commands should print `*_OK` or write a status file.
- If quoting, CRLF, PATH, or shell-boundary failures occur, record the invalid
  and corrected forms in the reliability protocol. Use
  `COMMAND_RELIABILITY_QUICKSTART.md` for current defaults and the longer
  protocol only for historical troubleshooting.

When docs and conversation conflict, prefer current GitHub evidence docs and
current cloud runtime state; do not resolve research status from the local
working tree. State uncertainty and cite the GitHub path or cloud path used.
