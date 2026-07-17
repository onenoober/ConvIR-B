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

- A PowerShell here-string added a UTF-8 BOM, so remote Bash did not activate
  strict mode and an outer success marker hid an inner failure. Do not pipe an
  ad hoc script across PowerShell/WSL/SSH. Commit the script with executable
  mode and use `convirctl.py remote-script`, which strips BOM/CRLF, validates
  strict mode, preserves the remote exit code, and requires its typed marker.
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
- 2026-07-17 invalid: dynamically JSON-quoted a multi-command `bash -lc` body
  inside PowerShell; the shell boundary produced an unmatched double quote
  before any repository command ran. Corrected once: invoke fixed Linux
  programs with literal argv through `wsl.exe --exec`; use a committed script
  when a command needs shell composition.
- 2026-07-17 incident: a schema-v4 start crossed the launch boundary and timed
  out after the runner had already produced a typed closeout, leaving no
  receipt and only an unactionable `inspect_once` label. Corrected in the
  unchanged six-tool surface: one repeat of the same sealed start performs a
  metadata-only inspection, recovers a receipt from exact bound runtime state,
  or proves and cleans only an untouched exact fresh workspace before retry.
- 2026-07-17 invalid: invoked bare `rg` after crossing from PowerShell into WSL;
  WSL PATH resolved the Codex WindowsApps binary and returned permission denied.
  Corrected once: use `wsl.exe --exec` with an available absolute Linux reader,
  here `/usr/bin/find`, `/bin/cat`, or `/usr/bin/sed`; never allow PATH fallback
  to select a Windows executable.
- 2026-07-17 invalid: passed an unquoted extended regular expression containing
  `|` through PowerShell, which parsed the alternatives as pipelines before WSL
  ran. Corrected once: stop that lookup and use a fixed absolute reader with
  literal argv; shell composition belongs only in a committed Bash script.
- 2026-07-17 invalid: a PowerShell here-string retained a UTF-8 BOM when piped
  through WSL and SSH, so remote Bash read a BOM-prefixed `set` token instead
  of the shell builtin. CR stripping alone is insufficient. Corrected once: strip the BOM
  on the first line and then CR before SSH, for example
  `sed '1s/^\xEF\xBB\xBF//' | tr -d '\r' | ssh convir-4090 'bash -s'`;
  require an explicit terminal `*_OK` marker and classify the BOM-bearing form
  as a command failure even if later read-only output was produced.
- 2026-07-17 invalid: embedded a Bash `for f ... "$f"` loop inside a
  PowerShell double-quoted `wsl.exe --exec /bin/sh -c` argument while checking
  documentation paths. PowerShell consumed the loop variable and WSL reported
  an unterminated quoted string before checking any file. Corrected once: call
  `wsl.exe --exec /usr/bin/test -f <literal-absolute-path>` separately for
  each fixed path; do not use a cross-shell variable loop for a finite path
  allowlist.
- 2026-07-17 invalid: relied on a PowerShell working directory for WSL Git,
  then embedded a quoted regular expression in `bash -lc`; Git inspected the
  wrong filesystem and PowerShell parsed the expression before WSL. Corrected
  once: pass the absolute WSL repository after `/usr/bin/git -C` and invoke
  fixed Linux readers through `wsl.exe --exec` with literal argv.
- 2026-07-17 invalid: passed Git revision syntax containing `^{commit}` through
  PowerShell, which interpreted the braces as a script block before WSL ran.
  Corrected once: use fixed-argv `git show -s --format=%H <sha>` when only
  commit existence/identity is needed; do not pass shell metacharacters across
  the Windows-to-WSL boundary.
