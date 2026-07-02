# Agent Instructions

## Hard Rules

- Local WSL = editing and syntax/compile-only checks. Do not run tests, smoke
  tests, training, evaluation, inference, demos, or runtime commands locally.
- Runtime validation runs only on `convir-4090` unless the user explicitly
  overrides a specific command. If unavailable, report it; do not fall back
  locally.
- Use explicit cloud Python paths, especially
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- GitHub `main` = durable compact evidence archive. Cloud = runtime/raw-output
  source. Local = editing/sync staging.
- Do not commit checkpoints, weights, datasets, images, arrays, archives, raw
  inference outputs, large per-image tables, selected-action tables, or raw
  feature tables by default.
- `github/codex/haze4k-official-arch-anchor` is immutable. New model-structure
  routes must branch from it.
- Use `experience_docx/` plus current git state as project memory; do not treat
  chat history as authoritative evidence.

## Read Budget

Read the smallest useful set; do not open all governance docs by default. Use
`rg` and targeted excerpts. Stop reading once the task is grounded.

| Task | Read |
| --- | --- |
| Experiment status/result/decision | `EXPERIMENT_INDEX.md`, then only the relevant family summary, route card, and evidence README/log dir |
| Cloud command, monitoring, sync, PowerShell/WSL/SSH | `COMMAND_RELIABILITY_PROTOCOL.md` |
| Training, smoke, eval, inference, post-run audit | `MODEL_RUN_OPERATIONS_PROTOCOL.md` |
| Evidence sync to GitHub | `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`, affected index/card/family/README |
| New Haze4K architecture/fine-tune route | `Haze4K_ARCH_FINETUNE_WORKFLOW.md`, partial-load/init/freeze rules |
| New experiment family/governance | Relevant sections only from `README.md`, governance/checklist/design/template docs |

## Sync Gates

- Sync because it is valuable, not because it is merely missing.
- Good candidates: route cards, compact READMEs, decisions, summary JSON,
  aggregate CSV, small config/status files, small reproducibility scripts,
  central index updates, family summaries, evidence README updates.
- High-value reasons: fixes a referenced GitHub evidence gap, changes route
  status/decision, records locked-test policy, documents a reproducible command,
  or closes an experiment.
- Use a clean worktree from `github/main`; stage explicit paths, never `git add .`.
- Before pushing evidence, check file types/sizes and run `git diff --check`.

## Cloud Gates

- Before launch, verify branch/commit, workspace, dataset, checkpoint, split,
  output root, tmux session, status file, command script, and locked-test policy.
- Do not overwrite active sessions, output dirs, or model names; inspect first.
- Every cloud run needs a durable command script, heartbeat/status, stdout/stderr
  capture, and compact evidence closeout.
- Distinguish infra/preflight/training/eval/scientific-gate failures explicitly.

## Command Reliability

- For PowerShell -> WSL -> SSH, prefer a small Bash script piped through WSL/SSH
  over fragile nested quoting.
- Monitoring/sync/audit commands should print `*_OK` or write a status file.
- If quoting, CRLF, PATH, or shell-boundary failures occur, record the invalid
  and corrected forms in the reliability protocol.

When docs and conversation conflict, prefer current repo docs and current git
state; state uncertainty and cite the path used.
