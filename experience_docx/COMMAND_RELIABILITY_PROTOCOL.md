# Command Reliability Protocol

Date: 2026-07-16

Validation evidence: `COMMAND_TRANSPORT_V1_VALIDATION.md` (28/28 cloud control
tests, zero model calls).

## One Selection Table

| Need | Only default | Do not use |
| --- | --- | --- |
| Windows -> WSL file, Git, or fixed program | `wsl.exe -d Ubuntu-22.04 --exec <absolute-program> <argv...>` | Windows Git on a WSL UNC path; `bash -lc`; nested quoting |
| Standard experiment plan/start/finish/evidence | the six bounded `convir-ops` v4 tools | generic SSH, dispatcher, watcher, or per-poll task |
| Cloud action not covered by MCP | committed, unchanged `.sh` file plus `convirctl.py remote-script --script <absolute-path>` | inline remote command, heredoc crossing shells, untracked/dirty script |
| Git/branch/SHA preflight | `convirctl.py git-state` | parsing human-formatted `git status` text |
| File identity | `convirctl.py sha256` | filename/mtime as identity |
| Result state | JSON, durable `status.txt`, typed closeout, and explicit `*_OK` marker | silence or terminal appearance as evidence |

GitHub carries branch, exact commit, tracked runner, rules, and compact evidence;
it never carries a command string. WSL Git owns WSL worktrees. Cloud owns raw
runtime state. File transfer uses an explicit allowlist plus SHA-256.

## Fixed Transport Contract

`experience_docx/tools/convirctl.py` has only `git-state`, `sha256`, and
`remote-script`. It uses argument arrays and fixed `/usr/bin/git`,
`/usr/bin/ssh`, `/bin/bash`, and host `convir-4090`. `remote-script` accepts only
an absolute workspace path to an unchanged Git-tracked `.sh`, reads its exact
committed blob, removes UTF-8 BOM and CRLF, requires the first executable line
to be `set -euo pipefail`, runs
`bash -n`, and sends the complete file on SSH stdin. It does not accept a remote
command string. Results are one JSON object with a marker and exit code; output
is capped at 64 KiB per stream.

From PowerShell, invoke the fixed WSL program directly. Example:

```powershell
wsl.exe -d Ubuntu-22.04 --exec /usr/bin/python3 /home/ubuntu/workspace/ConvIR-B/experience_docx/tools/convirctl.py git-state --repo /home/ubuntu/workspace/ConvIR-B --require-clean
```

## Finite Failure Rule

Quoting, CRLF/BOM, PATH, stdin, host-key, Git transport, and missing-marker
failures are command failures, never experiment evidence. Record the invalid
form and one canonical correction. If the same boundary class fails again, stop
that operation and report both forms. A timeout after a remote action begins is
`REMOTE_STATE_UNKNOWN`: inspect once and never retry blindly. Transport repair
cannot change experiment scope, evidence role, gate, data, or authorization.

Canonical corrections retained from prior incidents:

- PowerShell parsed an unescaped `|`, regex, `$()`, or quote: pass literal argv
  through `wsl.exe --exec`, or move the whole cloud body into a committed Bash
  file.
- Windows Git rejected a WSL UNC worktree: use WSL `/usr/bin/git -C
  /absolute/wsl/path`.
- a Windows executable leaked into WSL PATH: use the absolute Linux binary.
- SSH consumed wrapper stdin and skipped later markers: send one complete script
  as SSH stdin and return one structured result.
- 2026-07-16 invalid: cloud GitHub HTTPS ref read hit the fixed low-speed abort
  before validation. Corrected once: use the cloud's configured non-interactive
  GitHub SSH transport, shared clone, and one exact ref fetch.
- 2026-07-16 invalid: assumed `/usr/bin/rg` existed in WSL. Corrected once: test
  the explicit Linux path and use an already available fixed Linux reader when
  it is absent; never fall through to a Windows PATH executable.
- 2026-07-16 invalid: `convir-ops` wrote an internally generated launch body to
  `/tmp` and passed it to `convirctl remote-script`, whose external-operator
  contract correctly rejects scripts outside the workspace or Git. Corrected
  once: non-user-addressable MCP bodies use fixed `/usr/bin/ssh` argv and one
  complete stdin script with fixed host/shell, timeout, and 64 KiB stream caps.
  Manual actions still require an unchanged committed script through
  `convirctl remote-script`; no generic command surface was added.
- 2026-07-16 invalid: expected the current Codex task's MCP client to reconnect
  after normally terminating its uniquely identified old server process.
  Corrected once: do not hot-reload an active task's stdio transport. Fast-
  forward the dedicated registered worktree, then either let a new app/task
  session spawn the updated server or validate the exact registered executable
  with one isolated state directory and the unchanged six-tool interface.
