# Command Reliability Protocol

Date: 2026-07-27

This protocol is the single command-boundary contract. It covers the failure
classes observed during A1 and earlier runs without retaining incident-specific
shell recipes as operating instructions.

## Selection Table

| Need | Only default | Do not use |
| --- | --- | --- |
| Windows to WSL file, Git, or fixed program | wsl.exe -d Ubuntu-22.04 --exec followed by an absolute Linux program and literal argv | Windows Git on a WSL UNC path; bash -lc; nested quoting |
| Task/worktree binding | convirctl.py task-context --repo <repo> --cwd <cwd> | relying on the PowerShell or editor cwd |
| Repository read | convirctl.py repo-show, repo-list, or repo-search | cross-shell grep, sed, head, regex, or git-show pipelines |
| Standard plan/start/finish/progress/cancel/evidence | the six bounded convir-ops tools under stable protocol schema 4 | generic SSH, manual PID signals, dispatcher, watcher, or per-poll task |
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

## Progress Contract

Route entrypoints should call write_workload_progress from route_program_api for
milestones. It appends phase=workload and event=workload_progress with bounded
completed_units and total_units to status.txt. The MCP also accepts legacy
route-specific keys matching NAME_PROGRESS, so completed A1 routes remain
observable without preserving a route-specific parser.

Long engineering contracts use `write_contract_progress`, whose exact status
fields are phase, event, stage, completed_iterations and total_iterations. Any
extra field makes the milestone ineligible for MCP parsing, preventing metrics,
outcomes, data ids or scientific values from entering control telemetry. A
sealed finish result carrying retry-after/not-before is a command boundary:
another sealed call before not-before returns cached status, does not contact
the cloud and does not consume the finite finish-window budget. It does not
block `observation_mode=progress_only`, whose separately bounded and rate-
limited response contains only a token-safe stage, completed/total counts,
session activity, heartbeat age/source, snapshot time and an explicit cached
flag. Cached heartbeat age is never a current-health claim.

## Receipt-Bound Human Control

Human observation and cancellation are supported control-plane actions, not
exceptions to governance. `operator_action=cancel` accepts only the launch
receipt. The MCP derives and verifies route, run, commit, runner SHA-256,
workspace, output, closeout and tmux session, then verifies the exact lifecycle
process environment, command line, owner and Linux start ticks before signaling
it. PID-only input and manually assembled kill commands are invalid.

Cancellation first writes an identity-bound request, asks the lifecycle to
terminate its child process group and waits a bounded grace period. Forced
termination is allowed only after revalidating the same captured identities;
PID reuse or any mismatch fails closed. The terminal is
`CANCELLED_BY_OPERATOR / null / NONE`, is idempotent, and carries no scientific,
repair, promotion, archive or partial-evidence-reuse authorization. A transport
timeout is cancellation state unknown; repeat the same receipt-bound cancel
once for inspection/recovery, never issue an unrelated signal.

### 2026-07-20 WSL worktree Git boundary

Invalid form: invoking the Windows `git` selected by PowerShell while using a
`\\wsl.localhost\...` worktree path. Windows Git rejected the Linux-owned
worktree as dubious ownership before any repository mutation.

Corrected form: invoke `wsl.exe -d Ubuntu-22.04 --exec /usr/bin/git -C
<absolute-linux-worktree> ...` with literal argv. The final-slim control-plane
acceptance must retain a regression proving that the normal experiment path is
fully MCP-owned and never requires Windows Git against a WSL UNC path.

### 2026-07-20 WSL tool-path boundary

Invalid form: assume `/usr/bin/rg` exists inside WSL. The fixed exec failed
before searching or mutating anything because that absolute binary is absent.

Corrected form: use the available host `rg` for workspace searches, or a
literal PowerShell `Select-String` for Windows-owned Codex configuration. Never
substitute an unverified absolute executable path.

### 2026-07-20 WSL exec commit-message boundary

Invalid form: pass an unquoted multi-word commit message through PowerShell to
`wsl.exe --exec /usr/bin/git ... commit -m`; the words after the first were
delivered as pathspec arguments and Git made no commit.

Corrected form: use one literal argv token for the message at this boundary,
or invoke an argument-array transport that preserves the complete message.

### 2026-07-20 MCP runtime-activation workspace boundary

Invalid local probe: piping three JSON-RPC objects from a PowerShell array
directly into a WSL stdio MCP process returned only two responses and provided
no trustworthy activation result. Local runtime probing also exceeded the
repository's syntax-only local role, so it was stopped and not used as
evidence.

Canonical replacement: commit the bounded activation script unchanged and run
it on `convir-4090` through `convirctl.py remote-script`. The cloud script
starts a fresh stdio server, sends all three JSON-RPC requests from one Python
payload, requires exactly three response IDs, and emits an explicit
`CONVIR_OPS_V5_RUNTIME_ACTIVATION_OK` marker.

Invalid form: call `convir_git_status` against a fresh checkout under `/tmp`
during cloud-side stdio activation validation. The MCP correctly returned a
fail-closed structured tool error because the checkout was outside the trusted
project workspace; the first validation wrapper then read success-only fields
without checking `isError`.

First incomplete correction: placing the disposable validation checkout below
`/sda/home/wangyuxin/ConvIR-B/runtime/` still left the server's default local
workspace root at `/home/ubuntu/workspace`; the server again failed closed.

Canonical correction: keep the disposable checkout below the cloud project
runtime root and set `CONVIR_OPS_LOCAL_WORKSPACE_ROOT` for the validation
subprocess only to that checkout's parent directory. Production retains its
default `/home/ubuntu/workspace` boundary. The wrapper also asserts
`isError == false` before reading success-only structured fields. This is a
transport/workspace correction only and cannot change MCP code, scientific
data, metrics, thresholds, or historical evidence.

### 2026-07-21 PowerShell command-string metacharacter boundary

Invalid form: construct a PowerShell command string containing a nested
`bash -lc` body with a regular-expression alternation pipe. PowerShell parsed
the pipe before WSL received the intended literal command, so the read-only
inspection failed before repository access and made no filesystem change.

Canonical correction: use `convirctl.py repo-show`, `repo-list`, or
`repo-search` with literal argv for repository reads. When a fixed executable
must be invoked through WSL, use `wsl.exe --exec` with literal arguments and no
cross-shell metacharacters. A read failure never authorizes an alternate local
runtime command.

Invalid follow-up form: pass multiple quoted patterns containing nested square
brackets to one PowerShell Select-String command string. PowerShell split the
intended patterns into positional arguments, so the compatibility scan failed
before reading a trustworthy result and made no repository change.

Corrected form: issue one Select-String -SimpleMatch call per literal term, or
use convirctl.py repo-search for committed content. Treat a zero-match exit as
a successful read result and keep syntax/diff checks in separate commands.

### 2026-07-21 cloud unittest import-path boundary

Invalid form: invoke package-qualified unittest modules from the repository
root when legacy test modules import sibling tools and tests as top-level
modules. Collection stopped with ModuleNotFoundError before those test bodies
ran; this is FAILED_ENGINEERING and no acceptance conclusion is permitted.

Corrected form: keep the same committed test list and exact candidate commit,
but set PYTHONPATH to the committed experience_docx/tools and
experience_docx/tools/tests directories in the validation script. Use a new
commit-bound output directory, preserve the failed status/log, and rerun the
entire gate once.
