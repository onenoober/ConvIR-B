# Command Reliability Quickstart

Date: 2026-07-16

Status: current lightweight transport policy.

## Defaults

- Runtime host: `convir-4090`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Local WSL is editing and syntax/compile-only.
- Commands print `*_OK`/`*_FAILED` or write a status file.
- Quoting, CRLF, PATH, stdin, marker, and shell-boundary failures are
  `FAILED_COMMAND`, never experiment evidence.

## Transport Selection

| Task | Use |
| --- | --- |
| simple WSL Git/path read | direct argument array: `wsl.exe ... -- git -C ...` |
| bounded cloud route lifecycle/evidence | matching `convir-ops` schema-v3 tool |
| cloud command outside MCP boundary | `experience_docx/tools/convir_remote_script.sh <script>` |
| file transfer | explicit `scp`/`rsync`/`tar` paths |
| GitHub evidence sync | clean local GitHub-main worktree and explicit staged files |

Do not put simple Git/path operations in `bash -lc`. Do not build nested
PowerShell -> WSL -> SSH strings. A remote script defines its paths and
arguments and the wrapper strips BOM/CRLF, checks Bash syntax locally, uses
key-only SSH, and prints `CONVIR_REMOTE_SCRIPT_OK` only on remote success.

## Finite Recovery

1. Mark `FAILED_COMMAND` and discard partial output as evidence.
2. Apply one deterministic correction using the canonical form above.
3. If the same boundary class fails again, stop that operation with one blocker
   containing the two failed forms.
4. Read only the matching section of `COMMAND_RELIABILITY_PROTOCOL.md` when
   the correction is not obvious.

For an exact prelaunch MCP operation, retry once only when the typed result says
`failure_class=command_infra`, a retryable prelaunch phase is named, and
`runner_started=false`. Reuse the same plan, route, runner, data, thresholds
and output. `START_STATE_UNKNOWN` requires one inspection and never a blind
retry.

Command or implementation repair never creates scientific authorization or a
new model task.
