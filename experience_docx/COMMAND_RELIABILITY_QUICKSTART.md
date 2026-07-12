# Command Reliability Quickstart

Date: 2026-07-06

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
- Local WSL is for editing and syntax/static checks only.
- Every monitor, sync, or audit command should print a visible `*_OK` or
  `*_FAILED` marker, or write an explicit status file.
- Treat shell-boundary, quoting, CRLF, PATH, and silent-output failures as
  command failures, not scientific results.

## Transport Choices

| Task | Default transport | Read more only if |
| --- | --- | --- |
| PowerShell -> WSL local script | Write or pipe a small Bash script, strip BOM/CRLF, then run with WSL Bash. | The script needs nested quotes, loops, regex pipes, or heredocs. |
| WSL -> `convir-4090` remote script | Run `tools/convir_remote_script.sh <local-script>` from `experience_docx/`. | The remote command itself must consume stdin or needs file arguments. |
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

## Failure Handling

If a command fails because of transport, quoting, CRLF, PATH, stdin, or shell
boundary behavior:

1. label it `FAILED_COMMAND`;
2. do not interpret partial output as experiment evidence;
3. rerun only the affected operational step with a stable script-body pattern;
4. consult `COMMAND_RELIABILITY_PROTOCOL.md` for the matching historical
   failure class if the fix is not obvious.

## Archive Boundary

`COMMAND_RELIABILITY_PROTOCOL.md` is the detailed archive of known bad command
forms and corrected patterns. It is intentionally long and should not be read
by default during ordinary experiment planning or monitoring.
