# Command Reliability Protocol

Date: 2026-07-18

This protocol is the single command-boundary contract. It covers the failure
classes observed during A1 and earlier runs without retaining incident-specific
shell recipes as operating instructions.

## Selection Table

| Need | Only default | Do not use |
| --- | --- | --- |
| Windows to WSL file, Git, or fixed program | wsl.exe -d Ubuntu-22.04 --exec followed by an absolute Linux program and literal argv | Windows Git on a WSL UNC path; bash -lc; nested quoting |
| Task/worktree binding | convirctl.py task-context --repo <repo> --cwd <cwd> | relying on the PowerShell or editor cwd |
| Repository read | convirctl.py repo-show, repo-list, or repo-search | cross-shell grep, sed, head, regex, or git-show pipelines |
| Standard plan/start/finish/evidence | the six bounded convir-ops v4 tools | generic SSH, dispatcher, watcher, or per-poll task |
| Cloud action not covered by MCP | one committed, unchanged .sh through convirctl.py remote-script | inline SSH, heredoc across shells, untracked or dirty scripts |
| Git/branch/SHA preflight | convirctl.py git-state | parsing human-formatted status text |
| File identity | convirctl.py sha256 | filename or mtime |
| Result state | JSON, status.txt, typed closeout, and explicit *_OK marker | silence or terminal appearance |

## Fixed Transport Contract

convirctl.py uses argument arrays and fixed /usr/bin/git, /usr/bin/ssh,
/bin/bash, and host convir-4090. It has no arbitrary remote-command surface.
remote-script accepts only an absolute unchanged Git-tracked Bash file, removes
UTF-8 BOM and CRLF, requires the first executable line to be set -euo pipefail,
runs bash -n, preserves the remote exit code, and caps each output stream at
64 KiB. A timeout after remote execution begins is REMOTE_STATE_UNKNOWN and
allows inspection once, never a blind retry.

task-context fails closed when the requested cwd is not the requested worktree;
all writes must be explicitly bound to the local_repo path. The repo-show,
repo-list, and repo-search readers resolve a safe commit/ref and use literal
argv. repo-search accepts literal terms only; zero matches are a successful
empty result, not a command failure.

## Finite Recovery

Cross-shell metacharacters, BOM/CRLF, stdin loss, wrong WSL/Windows binary,
wrong worktree, silent outer markers, and unbounded verification wrappers are
command failures, never experiment evidence. Apply one deterministic correction
for a boundary class. A repeated class stops the operation with both the
invalid and canonical forms recorded. Transport repair cannot change data,
metrics, thresholds, seeds, budgets, evidence roles, or authorization.

The normal correction is to use the selection table, not to add another shell
layer. Internally generated MCP bodies use the fixed host and shell; manual
cloud actions use only a committed remote-script. Verification steps that
cross GitHub, SSH, and heartbeat boundaries must be separate bounded actions
with their own marker, timeout, and failure location.

### 2026-07-19 evidence-sync staging audit boundary

Invalid form: a PowerShell double-quoted `wsl ... bash -lc` command embedded
Bash command substitution such as `names=$(git diff --cached --name-only)`.
PowerShell expanded `$()` before WSL started, so the audit body reached Bash
malformed and no allow/deny decision was produced.

Corrected form: run `/usr/bin/git diff --cached --check` and
`/usr/bin/git diff --cached --name-only` as separate fixed WSL commands, then
apply the code-path and binary-extension predicates to the returned literal
name list in PowerShell. Require explicit `MAIN_EVIDENCE_STAGE_AUDIT_OK` only
after all predicates pass.

### 2026-07-19 route-identity mechanical rename boundary

Invalid form: generate one multi-file Perl program inside JavaScript and pass
it through a PowerShell-quoted `wsl.exe ... perl -e` command. The generated
program lost the loop-variable sigil before Perl execution and exited with
`Missing $ on loop variable`; no repository file was written.

Corrected form: invoke `/usr/bin/perl -pi -e` through fixed WSL argv once per
literal old/new identity pair, with repository files supplied as literal argv.
Use a delimiter absent from both literals (`#` for branch names containing
`/`); the default `/` delimiter caused one later pair to stop before writing.
Keep semantic changes in `apply_patch`, and require the staged route-ready gate
before commit.

## Progress Contract

Route entrypoints should call write_workload_progress from route_program_api for
milestones. It appends phase=workload and event=workload_progress with bounded
completed_units and total_units to status.txt. The MCP also accepts legacy
route-specific keys matching NAME_PROGRESS, so completed A1 routes remain
observable without preserving a route-specific parser.
