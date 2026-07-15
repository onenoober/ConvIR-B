# Command Reliability Quickstart

Date: 2026-07-12

Status: default lightweight command transport policy for new work.

## Purpose

Use this file first for PowerShell -> WSL -> SSH commands, cloud monitoring,
small syncs, and evidence archival commands. It contains the current default
patterns only. Read `COMMAND_RELIABILITY_PROTOCOL.md` only when a command fails,
when you need a historical failure mode, or when this quickstart is not specific
enough.

## Current Defaults

- Runtime host: `convir-4090`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- `convir-4090` SSH uses key-only noninteractive transport, a 15-second
  connection timeout, 10-second keepalives with two missed probes, and a
  10-minute per-host control connection.
- Local WSL is for editing and syntax/static checks only.
- Every monitor, sync, or audit command should print a visible `*_OK` or
  `*_FAILED` marker, or write an explicit status file.
- Treat shell-boundary, quoting, CRLF, PATH, and silent-output failures as
  command failures, not scientific results.
- Operations tools attach `failure_phase` to typed failures. Use the phase to
  retry only the affected engineering step; never infer a gate or metric result
  from a command failure.

## Transport Choices

| Task | Default transport | Read more only if |
| --- | --- | --- |
| PowerShell -> WSL local script | Write or pipe a small Bash script, strip BOM/CRLF, then run with WSL Bash. | The script needs nested quotes, loops, regex pipes, or heredocs. |
| Standard bounded `convir-4090` route operation in Codex | Automatically select the registered `convir-ops` MCP tool. | The task needs an operation outside its declared tool boundary. |
| WSL -> `convir-4090` remote script | Run `tools/convir_remote_script.sh <local-script>` from `experience_docx/`. | `convir-ops` is unavailable or the remote command must consume stdin or needs file arguments. |
| File sync to cloud | Use `tar`/`scp`/`rsync` on explicit files or directories. | The sync is large, incremental, or needs excludes. |
| Evidence sync to GitHub | Use a clean local `github/main` worktree; stage explicit text evidence paths. | Cloud GitHub credentials have been verified in the same task. |

## Required Markers

Use explicit markers so no-output success is not confused with a hang:

```bash
echo PRECHECK_OK
echo MONITOR_OK
echo SYNC_OK
echo COMMAND_FAILED
```

## Default WSL-To-Cloud Entry

Put the intended remote commands in a local Bash file, then run:

```bash
experience_docx/tools/convir_remote_script.sh /absolute/path/to/remote_body.sh
```

The wrapper strips a UTF-8 BOM and CRLF, checks Bash syntax locally, sends the
body to `ssh convir-4090 'bash -s'`, and prints `CONVIR_REMOTE_SCRIPT_OK` only
when the remote shell exits successfully. The script body should define all
remote paths and arguments itself; the wrapper deliberately performs no string
interpolation or general job orchestration.

Do not use this wrapper when the remote program needs the same stdin stream.
Transfer a durable script or input file first in that case.

## MCP-First Codex Entry

When `convir-ops` is registered, Codex should automatically select its bounded
schema-v2 tool that exactly matches the user task: manifest plan, authorized
start, bounded finish, compact-evidence manifest/fetch, or Git/evidence status.
Legacy preflight/launch/monitor schemas are not active tools. This avoids
rebuilding the same PowerShell -> WSL -> SSH command by hand and avoids adding
recovery-only schemas to every model context.

Tool selection is task-driven, not a timer or background scheduler. Launch
remains allowed only after the user requests execution and the route's typed
closeout and stage gates authorize it. The MCP cannot replace those checks,
GitHub evidence review, or explicit Git staging/commit/push. Use the script
wrapper for an operation that is not covered or when the MCP is unavailable.

The exact tool boundary is documented in `CONVIR_OPS_MCP.md`.

After updating the tracked MCP server, restart its host process and verify the
initialize response's source SHA-256 before the next route call. Evidence
manifest/fetch requests use the route's `repo_name` plus `workspace_id` so the
tool resolves the sealed hashed checkout rather than a guessed cloud path.

## Failure Handling

If a command fails because of transport, quoting, CRLF, PATH, stdin, or shell
boundary behavior:

1. label it `FAILED_COMMAND`;
2. do not interpret partial output as experiment evidence;
3. rerun only the affected operational step with a stable script-body pattern;
4. consult `COMMAND_RELIABILITY_PROTOCOL.md` for the matching historical
   failure class if the fix is not obvious.

For an exact R0/R1 operation, one bounded retry is permitted only when the
typed result says `failure_class=command_infra`, the phase is a retryable
transport/resource/evidence phase, and `runner_started=false`. Reuse the same
route, receipt or plan, runner, output, and thresholds. A timeout after the
launch boundary is `START_STATE_UNKNOWN` and requires inspection rather than a
blind retry.

## Archive Boundary

`COMMAND_RELIABILITY_PROTOCOL.md` is the detailed archive of known bad command
forms and corrected patterns. It is intentionally long and should not be read
by default during ordinary experiment planning or monitoring.
