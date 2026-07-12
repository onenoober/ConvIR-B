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
- New rule text must follow the repository document layers in
  `experience_docx/README.md`. Keep one canonical source per rule; other files
  should link to that source instead of copying the rule body. Do not mix
  runtime rules, design guidance, historical troubleshooting, branch cleanup, or
  experiment evidence in the same document section.

## Universal Route Workflow

This is the default execution flow for later reasoning and action. Use this
order for every new route, method, architecture, loss, selector,
teacher/distillation plan, or runtime experiment unless a stricter task-specific
protocol applies. Do not skip ahead to coding or launching because a previous
chat already seems to imply the answer.

1. Fact-source gate: first identify the authoritative sources for the task.
   For research state, use GitHub `main` or the named GitHub branch plus current
   cloud runtime state; use attachments only as task input. Treat local files as
   editable/sync staging unless the task is explicitly local editing.
2. Route-identity gate: decide whether this is a new route, a continuation, a
   rescue, an ablation, or an evidence sync. Record what is explicitly not
   allowed before work starts, especially forbidden continuations, selector
   probes, canary expansion, and locked-test access.
3. Resource-preflight gate: before any cloud launch, verify cloud branch/commit,
   workspace, explicit Python, dataset/split, checkpoint/teacher assets, output
   root, command script, status/log paths, tmux/session conflicts, and locked-test
   policy. If a required asset is missing, classify the blocker; do not silently
   substitute a different local or cloud source.
4. Metric-contract gate: define the exact baseline, sample/crop/split pairing,
   metric direction, gate thresholds, and comparison scope before running. Ensure
   before/after/base metrics are computed on the same data view; rerun rather
   than interpret if the metric contract is wrong. Formal gates must follow the
   canonical Gate Policy in `experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md`:
   classify the gate, record the threshold source and decision meanings, and do
   not interpret a failure beyond what that gate type tests.
5. Transport gate: use stable PowerShell -> WSL -> SSH transfer patterns from
   `COMMAND_RELIABILITY_QUICKSTART.md`. Prefer `tar`/`scp`/`rsync` or stable
   script bodies with explicit `*_OK` markers over ad hoc nested quoting; read
   the longer command protocol only for failed or unfamiliar boundary cases.
6. Stage-gate execution: run the smallest phase that answers the current
   question. Launch later phases only when the written previous gate authorizes
   them; never use locked test, canary expansion, or broad queues as a debugging
   shortcut.
7. Closeout and archive gate: after the last authorized phase, write the final
   decision into the route card, evidence README, closeout JSON, central index,
   and family summary as applicable. Keep runnable code on the route branch and
   sync only compact text evidence to GitHub `main` unless a separate promotion
   decision says otherwise.

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
| Experiment status/result/decision | GitHub `main`/named GitHub branch copy of `EXPERIMENT_INDEX.md`, then only the relevant family summary, route card, and evidence README/log dir |
| Cloud command, monitoring, sync, PowerShell/WSL/SSH | `COMMAND_RELIABILITY_QUICKSTART.md`; read `COMMAND_RELIABILITY_PROTOCOL.md` only for failed or unfamiliar command-boundary cases |
| Training, smoke, eval, inference, post-run audit | `MODEL_RUN_OPERATIONS_PROTOCOL.md` |
| Evidence sync to GitHub | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`, affected index/card/family/README |
| New Haze4K architecture/fine-tune route | `Haze4K_ARCH_FINETUNE_WORKFLOW.md`, partial-load/init/freeze rules |
| New experiment family/governance | Relevant sections only from `README.md`, governance/checklist/design/template docs |

## Sync Gates

- Sync only valuable compact text evidence, not raw artifacts or broad outputs.
- Use a clean `github/main` worktree, stage explicit paths, and follow
  `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` for path selection, checks, push, and
  verification.

## Cloud Gates

- Before launch, satisfy the resource-preflight and metric-contract gates in
  the Universal Route Workflow and `MODEL_RUN_OPERATIONS_PROTOCOL.md`.
- Do not overwrite active sessions, output dirs, or model names; inspect first.
- Every cloud run needs a durable command script, heartbeat/status, stdout/stderr
  capture, and compact evidence closeout.
- Distinguish infra/preflight/training/eval/scientific-gate failures explicitly.

## Command Reliability

- For PowerShell -> WSL -> SSH, prefer a small Bash script piped through WSL/SSH
  over fragile nested quoting.
- Monitoring/sync/audit commands should print `*_OK` or write a status file.
- If quoting, CRLF, PATH, or shell-boundary failures occur, record the invalid
  and corrected forms in the reliability protocol. Use
  `COMMAND_RELIABILITY_QUICKSTART.md` for current defaults and the longer
  protocol only for historical troubleshooting.

When docs and conversation conflict, prefer current GitHub evidence docs and
current cloud runtime state; do not resolve research status from the local
working tree. State uncertainty and cite the GitHub path or cloud path used.
