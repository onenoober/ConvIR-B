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

## Progress Contract

Route entrypoints should call write_workload_progress from route_program_api for
milestones. It appends phase=workload and event=workload_progress with bounded
completed_units and total_units to status.txt. The MCP also accepts legacy
route-specific keys matching NAME_PROGRESS, so completed A1 routes remain
observable without preserving a route-specific parser.

## 2026-07-20 Windows Git against a WSL UNC audit worktree

Invalid form:

```powershell
git show <ref>:<path>
```

when the PowerShell working directory is a `\\wsl.localhost\...` worktree.
Windows Git rejected the repository as dubious ownership before reading the
blob. This is `FAILED_COMMAND`, not experiment evidence.

Corrected form:

```powershell
wsl.exe -d Ubuntu-22.04 --exec /usr/bin/git -C /home/ubuntu/workspace/<worktree> show <ref>:<path>
```

Use fixed WSL argv or `convirctl.py repo-show`; never add a Windows Git
`safe.directory` exception for a WSL worktree.

## 2026-07-20 R14 remote-script outer timeout

Invalid form: invoking a bounded `convirctl.py remote-script` through a desktop
shell call whose outer timeout was only five seconds. The outer process ended
before `convirctl` could return its typed state, so the cloud state became
unknown even though the inner timeout was 3,600 seconds.

Corrected form: do not relaunch. Commit and run one fixed-path read-only
inspection script through `remote-script`, classify the existing output from
its status/log/closeout markers, then monitor or repair only from that observed
state. Any later remote-script invocation must give the desktop shell a timeout
longer than the bounded inner call.

## 2026-07-20 R14 evidence-sync branch check

Invalid form: require a freshly cloned branch HEAD to equal the earlier project-
memory commit exactly after the same branch had gained the committed sync
launcher. The fail-closed check returned `R14_SYNC_BASE_COMMIT_MISMATCH` before
copying or staging evidence.

Corrected form: in a fresh sync checkout, require the project-memory commit to
be an ancestor of the cloned audit-branch HEAD, then copy only the fixed compact
allowlist, validate suffixes and the staged diff, commit once and push the same
branch. This preserves lineage while allowing the committed launcher itself to
exist after the project-memory commit.

The next sync attempt reached the commit boundary but failed because the cloud
fresh clone had no repository or global author identity. The corrected form is
to set the existing local project identity (`Codex <codex@openai.com>`) only in
that dedicated sync repository, revalidate the already-staged compact allowlist,
then commit and push once. Do not set cloud global Git configuration.

## 2026-07-20 R15 independent probes and remote `stat` argv

Invalid forms observed while opening R15 were (1) collecting unrelated WSL and
SSH probes in one JavaScript `Promise.all`, which discarded successful outputs
when one SSH process timed out, and (2) passing `stat -c '%n %s'` through a
PowerShell/WSL/SSH command string, which split the format and made `stat` treat
`%s` as a path. These are `FAILED_COMMAND`, never repository, cloud-asset or
experiment evidence.

## 2026-07-20 R16 local WSL reader boundary

Invalid forms observed while opening R16 were (1) grouping three independent
WSL repository reads in one JavaScript \`Promise.all\`, so one malformed regex
reader discarded the other completed outputs, and (2) invoking the desktop
application's Windows \`rg\` path from WSL, which failed with \`Permission
denied\`. The regex reader also nested double quotes around a parenthesized
alternation inside \`bash -lc\`, so Bash received a syntactically invalid command.
These are \`FAILED_COMMAND\`, never repository or experiment evidence.

Corrected form: issue independent bounded fixed-argv reads, use PowerShell
\`Get-Content\` for literal local files, and use \`/usr/bin/git -C <worktree>
grep\` or the repository readers when WSL has no native \`rg\`. Do not use
\`bash -lc\`, cross-shell regex quoting, or the Windows application \`rg\` binary
inside WSL. Require the read command's explicit exit code before using its
output.

An additional invalid reader passed Git's unquoted
\`--format=%(refname:short)\` through PowerShell; PowerShell interpreted the
parenthesized atom before WSL received it. The corrected branch read is the
fixed-argv \`git branch -a\` form without formatting syntax. When formatted Git
output is essential, use \`convirctl.py git-state\` or a committed script.

Corrected form: keep independent identity probes in independent bounded calls
and use fixed `wsl.exe --exec /usr/bin/ssh ... <argv>` whenever the remote
command has no shell syntax. If a format string or pipeline is needed, place it
in a committed fixed-path script or avoid it in favor of separate `test`,
`sha256sum`, `git rev-parse` and `git status` probes, each with an explicit
marker. Do not relaunch or reinterpret an experiment because an outer probe
lost its output.
