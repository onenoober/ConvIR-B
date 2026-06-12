# Command Reliability Protocol

Date: 2026-06-04

Status: required workflow for avoiding repeated invalid commands in this repository.

## Purpose

This protocol records command forms that have already failed in this workspace
and the preferred forms that should be used instead. It is especially important
for monitoring cloud experiments from Windows PowerShell through WSL and then
over SSH to `convir-4090`.

## High-Priority Rule

Do not repeat a command form that failed because of quoting, CRLF, PATH, shell
boundary, or silent-output issues. Prefer stable script bodies with explicit
success markers over compact one-liners.

## Invalid Command Patterns To Avoid

### Escaped printf inside nested awk one-liners

Avoid embedding escaped quotes in an `awk` program inside a PowerShell-to-WSL
script body, especially inside command substitution:

```bash
numstat="$(git diff --cached --numstat | awk '{add+=$1; del+=$2} END {printf \"+%d -%d\", add, del}')"
```

Failure mode observed on 2026-06-12:

- the extra backslashes reached `awk`;
- `awk` reported `backslash not last character on line` and a syntax error;
- the surrounding audit continued, making the failure easy to miss.

Preferred form:

```bash
numstat="$(git diff --cached --numstat | python3 -c 'import sys
add=dele=0
for line in sys.stdin:
    parts=line.split()
    if len(parts)>=2 and parts[0].isdigit() and parts[1].isdigit():
        add+=int(parts[0]); dele+=int(parts[1])
print(f"+{add} -{dele}")')"
```

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

2026-06-12 recurrence:

Avoid assuming `rg` inside WSL is a Linux executable when the Windows app shim is
earlier on PATH:

```bash
rg -n "def build_net|fam_mode" Dehazing/ITS/models/ConvIR.py
```

Failure mode observed:

- WSL resolved `rg` to a WindowsApps Codex path;
- Bash returned `Permission denied` before any file search ran.

Corrected form:

```bash
if command -v /usr/bin/rg >/dev/null 2>&1; then
  /usr/bin/rg -n 'def build_net|fam_mode' Dehazing/ITS/models/ConvIR.py
else
  grep -nE 'def build_net|fam_mode' Dehazing/ITS/models/ConvIR.py
fi
```

2026-06-06 recurrence:

Avoid this form:

```powershell
wsl -d Ubuntu-22.04 bash -lc "cd /home/ubuntu/workspace/ConvIR-B && grep -n \"ConvIR-Dehaze-v1.6\|Evidence Inventory\" experience_docx/EXPERIMENT_INDEX.md"
```

Failure mode observed:

- PowerShell treated the `|Evidence` fragment as a pipeline boundary before
  WSL Bash received the intended command.

Corrected form:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
grep -nE 'ConvIR-Dehaze-v1\.6|Evidence Inventory' experience_docx/EXPERIMENT_INDEX.md || true
printf 'CORRECTED_GREP_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

2026-06-06 recurrence:

Avoid compact PowerShell-to-WSL SSH probes that try to embed Bash `$?` and
quoted remote commands inside one `bash -lc` string:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "set +e; timeout 10 ssh dehaze1 'printf DEHAZE1_SSH_OK\n'; printf 'alias_rc=%s\n' \"$?\""
```

Failure mode observed:

- nested quotes were closed at the wrong layer;
- WSL Bash received an unterminated command and returned
  `unexpected EOF while looking for matching '"'`.

Corrected form:

```powershell
$script = @'
set +e
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=8 dehaze1 'printf "%s\n" DEHAZE1_SSH_OK'
printf 'alias_rc=%s\n' "$?"
printf 'SSH_PROBE_DONE\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

Related recurrence:

Avoid embedding Bash command substitutions such as `$(basename "$f")` inside a
double-quoted PowerShell string passed to `wsl ... bash -lc`:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "for f in ~/.ssh/*.pub; do printf '%s ' "$(basename "$f")"; ssh-keygen -lf "$f"; done"
```

Failure mode observed:

- PowerShell attempted to run `basename` locally before WSL Bash received the
  script;
- Bash then received a malformed loop and returned a syntax error near `done`.

Corrected form:

```powershell
$script = @'
set -euo pipefail
for f in ~/.ssh/*.pub; do
  [ -f "$f" ] || continue
  printf '%s ' "$(basename "$f")"
  ssh-keygen -lf "$f"
done
printf 'PUBKEY_FINGERPRINTS_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

2026-06-06 recurrence:

Avoid invoking `ssh` from a WSL script without detaching stdin when the script itself is being piped into Bash:

```powershell
$script = @'
set -euo pipefail
ssh dehaze1 'printf "%s\n" OK'
printf 'LOCAL_DONE\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

Scope note:

- use `ssh -n ... 'remote command'` when the remote side is a single quoted
  command string and the wrapper stdin should stay local;
- do not add `-n` when the remote script is intentionally provided through
  stdin, for example `ssh host 'bash -s' < /tmp/remote_script.sh`, because
  `-n` will replace that stdin with `/dev/null` and the remote script will not
  run.

Failure mode observed:

- `ssh` inherited the wrapper stdin and consumed the remaining script body;
- lines after the `ssh` command never executed, so success markers and rc prints were missing.

Corrected form:

```powershell
$script = @'
set -euo pipefail
ssh -n dehaze1 'printf "%s\n" OK'
printf 'LOCAL_DONE\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

### Bash printf monitor formatting

2026-06-12 recurrence:

Avoid using a literal format string that starts with dashes in monitoring
helpers:

```bash
printf "--- %s ---\n" "$(basename "$f")"
```

Failure mode observed:

- `printf` treated the leading dashes as an option in the active shell context;
- the monitor exited early with `printf: --: invalid option`.

Corrected forms:

```bash
printf -- "--- %s ---\n" "$(basename "$f")"
printf '%s\n' "--- $(basename "$f") ---"
```


### Cloud Python and evidence copy assumptions

2026-06-10 recurrence:

Avoid assuming `python3` exists on `dehaze1` outside an activated environment:

```bash
ssh dehaze1 'cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor && python3 - <<"PY"\nprint("probe")\nPY'
```

Failure mode observed:

- `/tmp/cloud_py310_audit.sh: line 25: python3: command not found`;
- the cloud audit stopped before writing the code manifest.

Corrected form:

```bash
PY=/root/miniconda3/envs/py310/bin/python
ssh dehaze1 "cd /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor && $PY - <<'PY'
print('probe')
PY"
```

For project runtime commands, continue to prefer
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

2026-06-10 recurrence:

Avoid this scp form for copying the contents of a remote directory back into an
existing local evidence directory:

```bash
scp -r dehaze1:/tmp/cloud_py310_environment_20260610/. experience_docx/experiment_logs/cloud_py310_environment_20260610/
```

Failure mode observed:

- `error: unexpected filename: .`.

Corrected form:

```bash
rsync -a dehaze1:/tmp/cloud_py310_environment_20260610/ experience_docx/experiment_logs/cloud_py310_environment_20260610/
```

### PowerShell expansion of Bash variables inside WSL inline strings

2026-06-12 recurrence:

Avoid putting Bash variable assignments and expansions inside a PowerShell
command string passed directly to `wsl ... bash -lc`, for example:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'ANCHOR=$(git rev-parse --short ref); echo "anchor=$ANCHOR"'
```

Failure mode observed:

- PowerShell expanded `$ANCHOR`/`$V34` before Bash received the script;
- WSL Bash then saw an incomplete command and returned `syntax error:
  unexpected end of file`.

Corrected form:

```powershell
$script = @'
set -euo pipefail
ANCHOR=$(git rev-parse --short ref)
echo "anchor=$ANCHOR"
printf 'ANCHOR_PROBE_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
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

2026-06-06 recurrence:

Avoid this form even for simple local WSL checks:

```powershell
@'
set -euo pipefail
curl --version | head -n 1
'@ | wsl -d Ubuntu-22.04 -- bash
```

Failure mode observed:

- PowerShell inserted CRLF line endings;
- the script reached WSL Bash, but `head -n 1` received `1\r` and failed with
  `head: invalid number of lines: '1\r'`.

Corrected form:

```powershell
@'
set -euo pipefail
curl --version | head -n 1
printf 'LOCAL_WSL_CHECK_OK\n'
'@ | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
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
ssh dehaze1 '/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python script.py'
```

or inside a remote script:

2026-06-06 recurrence:

Avoid using a cloud-only interpreter path during local WSL static checks:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python -m py_compile some_file.py
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

Failure mode observed:

- the command ran inside local WSL, not on `dehaze1`;
- `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python` is a cloud path and returned
  `Permission denied` locally.

Corrected form:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
python3 -m py_compile some_file.py
printf 'LOCAL_PY_COMPILE_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

```bash
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
"$PY" script.py
```

2026-06-06 recurrence:

Avoid using `python3` inside remote SSH monitor or post-processing blocks even
when the main command already defines an explicit cloud interpreter path:

```bash
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
"$PY" summarize_queue.py > /tmp/summary.json
python3 - <<'PY'
import json
print(json.load(open('/tmp/summary.json'))['queue_progress_state'])
PY
```

Failure mode observed:

- the cloud image did not expose `python3` on PATH in that SSH session;
- the monitor exited after producing partial output and before the final
  success marker.

Corrected form:

```bash
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
"$PY" summarize_queue.py > /tmp/summary.json
"$PY" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('/tmp/summary.json').read_text(encoding='utf-8'))['queue_progress_state'])
PY
printf 'REMOTE_MONITOR_OK\n'
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

### PowerShell DOCX extraction with `Expand-Archive`

Avoid passing a `.docx` path directly to `Expand-Archive`:

```powershell
Expand-Archive -LiteralPath $docxPath -DestinationPath $dest
```

Failure mode observed:

- PowerShell rejected `.docx` as an unsupported archive extension even though
  DOCX is a ZIP container.

Corrected form:

```powershell
$zipPath = Join-Path $env:TEMP 'convir_report.zip'
Copy-Item -LiteralPath $docxPath -Destination $zipPath -Force
Expand-Archive -LiteralPath $zipPath -DestinationPath $dest -Force
```

### Avoid assigning to PowerShell `$HOME`

Avoid using `$home` or `$HOME` as a scratch variable in PowerShell:

```powershell
$home = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { 'C:\Users\Administrator\.codex' }
```

Failure mode observed:

- PowerShell treats `$HOME` case-insensitively as a read-only variable, so the
  assignment fails with `Cannot overwrite variable HOME`.

Corrected form:

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { 'C:\Users\Administrator\.codex' }
```

### Git `safe.directory` in copied cloud workspaces

Avoid assuming Git metadata is readable from a copied or root-owned cloud
workspace:

```bash
git -C /root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue status --short
```

Failure mode observed:

- Git reported `detected dubious ownership`, so launcher status could not
  record branch, commit, or working-tree state.

Corrected form for read-only evidence checks:

```bash
git config --global --add safe.directory /root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue
git -C /root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue branch --show-current
git -C /root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue rev-parse --short HEAD
```

### Running repo-root tools that import `Dehazing/ITS`

Avoid helper scripts that assume `os.getcwd()` is the import root:

```python
sys.path.insert(0, os.getcwd())
from data import test_dataloader
```

Failure mode observed:

- `eval_haze4k_checkpoint_compare.py` was launched from the repository root by
  the v1.8 queue, so `from data import test_dataloader` failed with
  `ModuleNotFoundError: No module named 'data'`.

Corrected form:

```python
TOOL_PATH = Path(__file__).resolve()
REPO_ROOT = TOOL_PATH.parents[2]
ITS_ROOT = REPO_ROOT / "Dehazing" / "ITS"
for path in (str(ITS_ROOT), str(REPO_ROOT), os.getcwd()):
    if path not in sys.path:
        sys.path.insert(0, path)
```

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

Do not combine `ssh -n` with a heredoc:

```bash
ssh -n dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "$REMOTE_ROOT"
REMOTE
```

Failure mode observed:

- `-n` redirected SSH stdin from `/dev/null`;
- the heredoc body was not delivered to remote `bash -s`;
- the wrapper continued to later local commands, and a following tar stream
  failed because the intended remote directory had never been created.

Preferred form:

```bash
ssh dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "$REMOTE_ROOT"
printf 'REMOTE_SETUP_OK\n'
REMOTE
```

Use `ssh -n` only for simple non-heredoc commands that must not read the
wrapper stdin.

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
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
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

## 2026-06-05 Local WSL wrapper quoting failure

Observed while inspecting the v1.5 evidence sync worktree from PowerShell: an
inline `wsl ... bash -lc '...'` command containing regex alternation pipes was
misparsed, so Bash received unquoted fragments such as `FullUDP` and
`haze4k_fulludp` as shell commands.

Invalid form:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc 'rg -n "v1\.5|FullUDP|UDPNet|haze4k_fulludp" experience_docx/EXPERIMENT_INDEX.md'
```

Corrected form:

```powershell
@'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
rg -n "v1\.5|FullUDP|UDPNet|haze4k_fulludp" experience_docx/EXPERIMENT_INDEX.md || true
'@ | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

For PowerShell-to-WSL commands with regex pipes, nested quotes, or multiple
commands, prefer the here-string wrapper even when no SSH hop is involved.

## 2026-06-05 WSL PATH leaking Codex Windows `rg`

Observed while reading v1.5 sync evidence from WSL: `command -v rg` resolved to
the Codex Windows app resource path under `/mnt/c/Program Files/.../rg`, and
Bash failed with `Permission denied`.

Invalid form:

```bash
rg -n "GitHub|text evidence|checkpoint|weights|BRANCH|sync|push" experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md
```

Corrected form:

```bash
grep -En "GitHub|text evidence|checkpoint|weights|BRANCH|sync|push" experience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md
```

When `rg` resolves to a Windows app resource from inside WSL, fall back to
`grep -En` or install/use a native WSL ripgrep binary before continuing.

## 2026-06-05 `dehaze1` endpoint update

The `dehaze1` SSH alias was updated after the cloud server was replaced:

```sshconfig
Host dehaze1 seetacloud
  HostName connect.bjb1.seetacloud.com
  Port 42371
  User root
  IdentityFile ~/.ssh/id_ed25519_seetacloud
  IdentitiesOnly yes
```

Validation marker from the new server:

```text
DEHAZE1_ALIAS_CONNECT_OK
```

The previous port `49601` is obsolete for current v1.5 work. The official
ConvIR+UDP Haze4K checkpoint is now expected at:

```text
/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt
```

## 2026-06-05 Nested single-quote SSH monitor failure

Observed while monitoring the v1.5 official eval: a compact `ssh ... 'bash -lc
'\''...'\'''` command mixed nested single quotes with PowerShell here-string
transport and failed before running the remote monitor.

Invalid form:

```bash
ssh dehaze1 'bash -lc '\''set -euo pipefail; E=/path; tail -n 40 "$E/status.txt"; echo OK'\'''
```

Corrected form:

```bash
ssh dehaze1 'bash -s' <<'REMOTE'
set -euo pipefail
E=/path
tail -n 40 "$E/status.txt"
printf 'REMOTE_MONITOR_OK\n'
REMOTE
```

For monitor commands with variables and quoted paths, use the quoted heredoc
form instead of nested single-quote `bash -lc` one-liners.

## 2026-06-06 WSL inline alternation and Windows `rg` recurrence

Observed while preparing the v1.8 execution queue: an inline WSL search with
regex alternation was split by the wrong shell layer, so fragments such as
`val_regular`, `val_hard`, `active_adapters`, and `partial` were executed as
commands. A follow-up `command -v rg` check also resolved to the Codex Windows
app resource under `/mnt/c/Program Files/.../rg`, which failed with
`Permission denied`.

Invalid form:

```powershell
wsl -d Ubuntu-22.04 bash -lc 'cd /home/ubuntu/workspace/ConvIR-B && grep -RIl "split_json|val_regular|val_hard|active_adapters|partial" Dehazing/ITS experience_docx/tools'
```

Corrected form:

```powershell
$script = @'
set -euo pipefail
cd /home/ubuntu/workspace/ConvIR-B
grep -RInE 'split_json|val_regular|val_hard|active_adapters|partial' Dehazing/ITS experience_docx/tools || true
printf 'LOCAL_SEARCH_OK\n'
'@
$script | wsl -d Ubuntu-22.04 -- bash -lc "tr -d '\r' | bash"
```

Inside WSL wrappers, prefer native `grep -RInE` unless a native WSL `rg` binary
is already verified. Do not rely on `command -v rg` alone when Windows PATH
entries are visible inside WSL.

## 2026-06-06 Bash `printf` with format starting in dashes

Observed while monitoring the v1.8 execution queue from a PowerShell -> WSL ->
SSH heredoc wrapper: a remote Bash marker used the marker text itself as the
`printf` format and the format began with dashes.

Invalid form:

```bash
printf '--- status_tail ---\n'
```

Failure mode observed:

- Bash treated the dash-prefixed format as an option in this shell context;
- the monitor exited before printing status tails or the final success marker.

Corrected form:

```bash
printf '%s\n' '--- status_tail ---'
```

For all marker lines that start with `-`, use an explicit string format such as
`printf '%s\n' "$marker"` rather than making the marker the format string.

## 2026-06-06 Remote monitor helper must use explicit cloud Python

Observed while checking the v1.8 execution queue: the main monitor and progress
refresh succeeded because they used the explicit cloud interpreter, but an
additional convenience snippet inside the same remote heredoc used bare
`python3` for lightweight JSON printing and failed with `python3: command not
found`.

Invalid form:

```bash
python3 - <<'PY'
import json
...
PY
```

Corrected form:

```bash
"$PY" - <<'PY'
import json
...
PY
```

Inside cloud monitor/audit helpers, use the already-declared explicit runtime
such as `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python` or `"$PY"` for all
inline Python snippets as well; do not assume `python3` exists on PATH.

## 2026-06-11 convir-4090 DTA-v3 monitoring notes

Observed while monitoring the DTA-v3 Phase A run on `convir-4090`: the cloud
environment does not guarantee a bare `python` executable on PATH, and compact
PowerShell -> WSL -> SSH heredocs with nested quotes are fragile.

Invalid forms:

```bash
python - <<'PY'
...
PY
```

```powershell
wsl ... bash -lc "ssh convir-4090 'bash -lc "python - <<'PY' ..."'"
```

Corrected form:

```bash
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
ssh convir-4090 'bash -s' < local_monitor_script.sh
```

## 2026-06-12 local syntax check Python name

Avoid assuming `python` exists in the local WSL editing environment:

```bash
python -m py_compile experience_docx/tools/eval_haze4k_checkpoint_compare.py
```

Failure mode observed:

- local WSL returned `python: command not found`;
- this was a static/syntax-only check, not model runtime.

Corrected form:

```bash
python3 -m py_compile experience_docx/tools/eval_haze4k_checkpoint_compare.py
```

## 2026-06-12 remote monitor helper unbound variable

Avoid using `"$PY"` inside a remote monitor heredoc unless the monitor script
defines it first:

```bash
"$PY" - <<'PY' "$json_path"
...
PY
```

Failure mode observed:

- the remote monitor used `set -u` and exited with `PY: unbound variable`;
- the training/evaluation jobs were unaffected, but the monitor did not print
  the intended JSON summaries.

Corrected form:

```bash
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
"$PY" - "$json_path" <<'PY'
...
PY
```

Use `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python` or `"$PY"` for
all cloud Python snippets, and prefer a small local script piped to remote
`bash -s` over deeply nested inline heredocs.

2026-06-12 recurrence:

A cloud probe again used `ssh convir-4090 '...'` inside a piped WSL Bash script
without `-n`. The remote command succeeded, but SSH consumed the remaining local
script body, so `SSH_RC` and the final local success marker did not print.

Invalid form:

```bash
ssh convir-4090 'printf "HOST_OK\n"'
printf 'SSH_RC=%s\n' "$?"
```

Corrected form:

```bash
ssh -n convir-4090 'printf "HOST_OK\n"'
printf 'SSH_RC=%s\n' "$?"
```

Use `ssh -n` for all single-command remote probes launched from a piped local
script; omit `-n` only when the remote command intentionally reads a script from
stdin.

### `convir-5090` tmux assumption

2026-06-12 DTA-v3.4 launch note: do not assume `tmux` exists on
`convir-5090`. The first depth-cache launcher used `tmux new-session` and failed
with `bash: line 45: tmux: command not found`.

Corrected pattern for `convir-5090` background jobs:

```bash
nohup bash "$SCRIPT" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
```

Record the PID file and monitor with `kill -0 $(cat "$PIDFILE")` plus status
markers instead of tmux session checks.

### 2026-06-12 `convir-5090` Depth Cache Recovery

The first DTA-v3.4 depth-cache run on `convir-5090` failed because Hugging Face
requests returned `[Errno 101] Network is unreachable`. The corrected recovery
was to copy the already generated Depth Anything cache from `convir-4090`:

```bash
ssh -n convir-4090 'cd /sda/home/wangyuxin/ConvIR-B/depth_cache && tar -cf - depth_anything_v2_small_hf' \
  | ssh convir-5090 'cd /home/caozhiyang/ConvIR-B/depth_cache && tar -xf -'
```

Use `ssh -n` for the source side when streaming tar from a command script; the
first attempt without `-n` allowed the source SSH command to consume the wrapper
stdin, so the local trailing success marker did not print even though extraction
succeeded.

When `set -o pipefail` is active, avoid `find ... | head` as a success-checking
pipeline because `find` can receive SIGPIPE after `head` exits. Use `python`,
`sed -n`, or disable pipefail locally for preview-only listings.
