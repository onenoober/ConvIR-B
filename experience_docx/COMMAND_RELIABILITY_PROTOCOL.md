# Command Reliability Protocol

Date: 2026-06-04

Status: required workflow for avoiding repeated invalid commands in this repository.

## Purpose

This protocol records command forms that have already failed in this workspace
and the preferred forms that should be used instead. It is especially important
for monitoring cloud experiments from Windows PowerShell through WSL and then
over SSH to `dehaze1`.

## High-Priority Rule

Do not repeat a command form that failed because of quoting, CRLF, PATH, shell
boundary, or silent-output issues. Prefer stable script bodies with explicit
success markers over compact one-liners.

## Invalid Command Patterns To Avoid

### PowerShell to WSL inline regex pipes

Avoid inline commands where PowerShell, WSL Bash, and regex pipes all appear in
one string, for example:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'cd /repo && rg -n "v1.4|UDP-Lite" file.md'
```

Failure mode observed:

- the `|` regex was split or interpreted at the wrong shell layer;
- `rg` was resolved to a Windows app path inside WSL and failed with permission
  errors;
- fragments of the command were executed as separate commands.

Preferred forms:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
grep -En 'v1\.4|UDP-Lite' experience_docx/EXPERIMENT_INDEX.md || true
printf 'LOCAL_DOC_CHECK_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

or, when `rg` is required, resolve it inside WSL:

```bash
command -v rg >/dev/null && rg -n 'v1\.4|UDP-Lite' experience_docx/EXPERIMENT_INDEX.md || grep -En 'v1\.4|UDP-Lite' experience_docx/EXPERIMENT_INDEX.md
```

### PowerShell here-string to WSL heredoc without CR stripping

Avoid sending a PowerShell here-string directly to a WSL script that contains a
heredoc:

```powershell
$script | wsl -d Ubuntu-22.04 -- bash
```

Failure mode observed:

- PowerShell inserted CRLF line endings;
- the remote heredoc delimiter became `REMOTE\r`;
- the remote body executed, but the local wrapper returned nonzero with
  `wanted 'REMOTE'` or `REMOTE\r: command not found`.

Preferred form:

```powershell
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

For nested SSH scripts, use a quoted heredoc after CR stripping:

```powershell
$script = @'
set -euo pipefail
ssh dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
printf 'remote_time=%s\n' "$(date -Is)"
printf 'REMOTE_STATUS_OK\n'
REMOTE
printf 'LOCAL_WRAPPER_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

### Assuming `python` exists on the cloud server

Avoid:

```bash
ssh dehaze1 'python script.py'
```

Failure mode observed:

- `python: command not found` in cloud helper scripts.

Preferred form:

```bash
ssh dehaze1 '/root/miniconda3/envs/convir-cu128/bin/python script.py'
```

or inside a remote script:

```bash
PY=/root/miniconda3/envs/convir-cu128/bin/python
"$PY" script.py
```

### Silent monitoring commands

Avoid commands that can succeed with no visible output:

```bash
ssh dehaze1 'tmux has-session -t run_name'
```

Failure mode observed:

- a successful no-output command looked like a hang or invalid command.

Preferred form:

```bash
ssh dehaze1 'if tmux has-session -t run_name 2>/dev/null; then echo run_name=ACTIVE; else echo run_name=NOT_ACTIVE; fi; echo MONITOR_OK'
```

Every monitoring or sync command should print a final marker such as:

- `REMOTE_STATUS_OK`
- `EVIDENCE_SYNC_OK`
- `MONITOR_SCRIPT_OK`
- `COMMIT_AND_PUSH_OK`

### Non-heredoc SSH commands consuming the WSL wrapper stdin

Avoid running a plain `ssh host "cmd"` inside a PowerShell here-string wrapper
before later local commands:

```bash
ssh dehaze1 "mkdir -p '$REMOTE_ROOT'"
tar -cf - files | ssh dehaze1 "tar -C '$REMOTE_ROOT' -xf -"
```

Failure mode observed:

- the first `ssh` command consumed the remaining wrapper stdin;
- later sync/verification commands were never executed;
- the wrapper ended without the expected success marker, making the operation
  look like a silent success.

The same applies to remote-to-local tar streams in wrapper scripts:

```bash
ssh dehaze1 "tar -C '$REMOTE_ROOT' -cf - evidence/path" | tar -xf -
python3 parse_evidence.py
```

If `ssh` is not run with `-n`, it can consume the remaining wrapper stdin before
the local parse step runs.

Preferred forms:

```bash
ssh -n dehaze1 "mkdir -p '$REMOTE_ROOT'"
tar -cf - files | ssh dehaze1 "tar -C '$REMOTE_ROOT' -xf -"
ssh -n dehaze1 "tar -C '$REMOTE_ROOT' -cf - evidence/path" | tar -xf -
```

or use a quoted heredoc for all remote setup/verification:

```bash
ssh dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "$REMOTE_ROOT"
printf 'REMOTE_SETUP_OK\n'
REMOTE
```

### Syncing CR-suffixed filenames from broken wrappers

Avoid blindly staging files after a failed CRLF heredoc wrapper. Failure mode
observed:

- a duplicate path such as `status.txt\r` was created and staged.

Preferred cleanup before staging evidence:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('experience_docx/experiment_logs')
for p in list(root.rglob('*')):
    if '\r' in str(p):
        if p.is_file():
            p.unlink()
        else:
            raise SystemExit(f'unexpected CR path: {p!r}')
print('CR_PATH_CLEAN_OK')
PY
```

## Standard Cloud Monitor Template

Use this template for future training and post-eval checks:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
ssh dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
EVID=/root/autodl-tmp/workspace/<remote-workspace>/experience_docx/experiment_logs/<route_id>
PY=/root/miniconda3/envs/convir-cu128/bin/python
printf 'remote_time=%s\n' "$(date -Is)"
for s in <train_tmux> <post_tmux>; do
  if tmux has-session -t "$s" 2>/dev/null; then
    printf '%s=ACTIVE\n' "$s"
  else
    printf '%s=NOT_ACTIVE\n' "$s"
  fi
done
[ -f "$EVID/status.txt" ] && tail -n 80 "$EVID/status.txt" || printf 'status=MISSING\n'
printf 'REMOTE_MONITOR_OK\n'
REMOTE
rsync -a dehaze1:/root/autodl-tmp/workspace/<remote-workspace>/experience_docx/experiment_logs/<route_id>/ experience_docx/experiment_logs/<route_id>/
printf 'EVIDENCE_SYNC_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

## Standard GitHub Evidence Commit Template

Use this template when cloud evidence is complete:

```bash
git add AGENTS.md experience_docx/COMMAND_RELIABILITY_PROTOCOL.md experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md experience_docx/EXPERIMENT_INDEX.md experience_docx/family_summaries/<family>_summary.md experience_docx/experiment_cards/<card>.md experience_docx/experiment_logs/<route_id>
git diff --cached --check
git diff --cached --name-only | grep -Ei '\.(pkl|pth|pt|ckpt|onnx|png|jpg|jpeg|bmp|gif|webp|npy|npz|mat|zip|tar|gz|7z|rar)$' && exit 1 || true
git commit -m "Sync <route> evidence"
git push github HEAD:$(git branch --show-current)
```

If unrelated worktree changes exist, stage only the intended files and verify
with `git diff --cached --name-only` before committing.
