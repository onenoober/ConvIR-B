# Agent Instructions

## Cloud And Local Execution

- Default cloud server for this repository: `convir-4090` (`ssh convir-4090`).
- `convir-4090` is the only default runtime host; use the SSH key and endpoint configured in `~/.ssh/config`.
- Highest-priority project rule: the local WSL checkout is for editing and compile/syntax-only checks.
- Do not run tests, smoke tests, training, evaluation, inference, demos, or project runtime commands locally.
- Run all tests, including smoke tests, training, evaluation, and any execution/runtime validation, on `convir-4090` unless the user explicitly overrides this rule for a specific command.
- If runtime validation is needed, sync the code to `convir-4090` and run it there; if the cloud server is unavailable, report that instead of falling back to local execution.


## GitHub Evidence Sync

- After any cloud experiment, training, evaluation, audit, or post-run watcher finishes, syncing text evidence to GitHub is the first-priority archival step.
- Before considering a completed cloud run closed, sync the cloud evidence back into `experience_docx/`, update the route card, central index, family summary, and evidence README, then commit and push the text evidence to GitHub unless the user explicitly says not to.
- Treat GitHub as the primary durable share/read location for completed experiment evidence; the cloud server copy is a runtime source, not the final evidence archive.
- Do not commit checkpoints, model weights, datasets, images, arrays, archives, or raw inference outputs by default; sync only text evidence and small structured artifacts allowed by `experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.
- If GitHub push is unavailable, report the failure and the exact local evidence paths instead of treating the cloud-only copy as synced.

## Official Architecture Anchor And Clean Route Gate

- Mandatory anchor branch for future Haze4K ConvIR-B architecture work: `github/codex/haze4k-official-arch-anchor`. Treat it as protected even if GitHub UI branch protection is not configured.
- On the anchor branch, do not edit model architecture, runtime entrypoints, data loaders, losses, evaluation tooling, checkpoint-selection logic, gates, selectors, adapters, or experiment variants. Documentation, command-reliability fixes, and text evidence sync are the only normal allowed changes.
- Any architecture, loss, training-policy, data-policy, selector/gate, adapter, or fine-tuning experiment MUST start from a new `codex/<route>` branch or isolated worktree created from `codex/haze4k-official-arch-anchor`. Do not start from a dirty worktree or a previous failed route unless the route card explicitly labels it as a continuation.
- Before the first code edit or cloud run on a new route, create or update the route card with the anchor commit, branch name, remote workspace, checkpoint path/hash, strict or partial checkpoint-load rule, locked-test policy, output root, command script path, gates, and evidence sync plan.
- A route that modifies the official architecture must prove checkpoint initialization behavior before fine-tuning: either strict `haze4k-base.pkl` load for unchanged modules or an explicit partial-load/new-module initialization contract recorded in the route card.
- If a request would modify the anchor branch in a blocked way, stop before editing or running, state the blocker, and create or switch to a route branch first.

## Command Reliability

- For multi-hop commands involving PowerShell, WSL, and `ssh convir-4090`, read and follow `experience_docx/COMMAND_RELIABILITY_PROTOCOL.md` before running the command.
- Avoid complex inline PowerShell-to-WSL-to-SSH one-liners with nested quotes, regex pipes, or heredocs; write a small Bash script body and pipe it through `wsl ... bash -lc "tr -d '\r' | bash"` instead.
- Every monitoring, sync, or audit command should print an explicit `*_OK` success marker or write a status file so a successful no-output command is not mistaken for a hang.
- Use explicit runtime paths for cloud Python, especially `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`, instead of assuming `python` is on PATH.
- If a command fails from quoting, CRLF, PATH, or shell-boundary issues, record the invalid form and the corrected form in the command reliability protocol before continuing.

## Model Run Operations

- For any model training, smoke test, evaluation, inference, post-run audit, or runtime validation, read and follow `experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md` before launching or monitoring the run.
- Before launching a cloud run, verify branch/commit, remote workspace path, data path, checkpoint path, split file, output root, tmux session name, status file, command script, and locked-test policy.
- Do not relaunch or overwrite an active run with the same session, output directory, or model name; inspect tmux/status/checkpoints first and either resume explicitly or create a new route/run id.
- Every cloud run must have a durable command script, a `status.txt` or equivalent heartbeat log, stdout/stderr log capture, and post-run evidence sync back to `experience_docx/`.
- Treat smoke/preflight failures, training failures, eval failures, and scientific gate failures as different states; record the state explicitly instead of retrying with changed scope silently.

## Project Memory And Evidence Authority

- Treat `experience_docx/` as the repository's authoritative project memory for experiment state, governance, route decisions, and evidence locations.
- Do not use chat history as the authoritative source for experiment results, route status, decisions, commands, or evidence; verify against repository documents first.
- For any experiment, route, training, evaluation, result-summary, or decision task, read `experience_docx/EXPERIMENT_INDEX.md` first.
- After the index, read the relevant `experience_docx/family_summaries/` file when reopening or extending a route family.
- Then open the corresponding route card under `experience_docx/experiment_cards/`, and inspect the matching evidence directory under `experience_docx/experiment_logs/` before making claims or planning follow-up work.
- For branch evidence syncs, read and follow `experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.
- For new or reorganized experiments, read the relevant `experience_docx/` governance docs first, especially `README.md`, `OFFICIAL_ARCH_ANCHOR_POLICY.md`, `CONVIR_B_EXECUTION_GUIDE.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `MODEL_EXPERIMENT_START_CHECKLIST.md`, `ROUTE_DESIGN_FRAMEWORK.md`, and `EXPERIMENT_CARD_TEMPLATE.md`.
- When documentation and conversation conflict, prefer `experience_docx/` and current git state; state any uncertainty and cite the file path used.
