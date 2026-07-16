#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <candidate-commit> <powershell-7.4.6-linux-x64.tar.gz>" >&2
  exit 2
fi

candidate=$1
pwsh_archive=$2
repo_url=git@github.com:onenoober/ConvIR-B.git
pwsh_sha256=6f6015203c47806c5cc444c19d8ed019695e610fbd948154264bf9ca8e157561
cloud_python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
run_root="/tmp/convir-experiment-control-validation-${candidate:0:12}"
checkout="$run_root/repository"
pwsh_root="$run_root/powershell-7.4.6"
status_path="$run_root/status.txt"
stdout_path="$run_root/stdout.log"
stderr_path="$run_root/stderr.log"
summary_path="$run_root/summary.txt"

if [[ ! $candidate =~ ^[0-9a-f]{40}$ ]]; then
  echo "CONTROL_VALIDATION_FAILED invalid_candidate=$candidate" >&2
  exit 2
fi
if [[ ! -f $pwsh_archive ]]; then
  echo "CONTROL_VALIDATION_FAILED missing_pwsh_archive=$pwsh_archive" >&2
  exit 2
fi
if [[ -e $run_root ]]; then
  echo "CONTROL_VALIDATION_FAILED existing_run_root=$run_root" >&2
  exit 2
fi

mkdir -p "$run_root"
printf 'state=RUNNING\ncandidate=%s\nmodel_calls=0\n' "$candidate" > "$status_path"

cleanup() {
  rc=$?
  rm -rf "$checkout" "$pwsh_root"
  rm -f "$pwsh_archive"
  if [[ $rc -ne 0 ]] && ! grep -q '^state=FAILED$' "$status_path" 2>/dev/null; then
    printf 'state=FAILED\ncandidate=%s\nmodel_calls=0\nexit_code=%s\n' "$candidate" "$rc" > "$status_path"
  fi
  exit "$rc"
}
trap cleanup EXIT

observed_pwsh_sha256=$(sha256sum "$pwsh_archive" | awk '{print $1}')
if [[ $observed_pwsh_sha256 != "$pwsh_sha256" ]]; then
  echo "CONTROL_VALIDATION_FAILED pwsh_sha256=$observed_pwsh_sha256" >&2
  exit 1
fi

git clone --quiet --no-checkout "$repo_url" "$checkout"
git -C "$checkout" checkout --quiet --detach "$candidate"
observed_candidate=$(git -C "$checkout" rev-parse HEAD)
if [[ $observed_candidate != "$candidate" ]]; then
  echo "CONTROL_VALIDATION_FAILED candidate_mismatch=$observed_candidate" >&2
  exit 1
fi

mkdir -p "$pwsh_root"
tar -xzf "$pwsh_archive" -C "$pwsh_root"
chmod +x "$pwsh_root/pwsh"
pwsh_version=$($pwsh_root/pwsh -NoLogo -NoProfile -Command '$PSVersionTable.PSVersion.ToString()')
if [[ $pwsh_version != "7.4.6" ]]; then
  echo "CONTROL_VALIDATION_FAILED pwsh_version=$pwsh_version" >&2
  exit 1
fi

dispatcher="$checkout/experience_docx/tools/dispatch_agent_task.ps1"
contract_test="$checkout/experience_docx/tools/test_dispatch_agent_task.ps1"
set +e
"$pwsh_root/pwsh" -NoLogo -NoProfile -File "$contract_test" \
  -DispatcherPath "$dispatcher" \
  -RepositoryLinuxPath "$checkout" \
  >"$stdout_path" 2>"$stderr_path"
test_rc=$?
set -e
if [[ $test_rc -ne 0 ]]; then
  echo "CONTROL_VALIDATION_FAILED contract_test_rc=$test_rc" >&2
  tail -n 40 "$stderr_path" >&2 || true
  exit "$test_rc"
fi

"$cloud_python" - "$stdout_path" "$checkout/experience_docx/CONVIR_OPS_MCP.md" "$summary_path" <<'PY'
import json
import re
import sys
from pathlib import Path

stdout_path, mcp_path, summary_path = map(Path, sys.argv[1:])
raw = stdout_path.read_text(encoding="utf-8")
marker = "DISPATCHER_DRY_RUN_TESTS_OK"
if marker not in raw:
    raise SystemExit("missing dispatcher completion marker")
payload = json.loads(raw[: raw.rfind(marker)].strip())
if payload.get("status") != "PASS" or payload.get("model_calls") != 0:
    raise SystemExit("dispatcher contract status/model_calls mismatch")
cases = payload.get("cases", [])
if len(cases) != 27 or any(case.get("decision") != "PASS" for case in cases):
    raise SystemExit(f"dispatcher case matrix mismatch: {len(cases)}")
required = {
    "r1_luna",
    "mismatched_r1_evidence",
    "execute_requires_explicit_opt_in",
    "explicit_opt_in_reaches_platform_gate",
    "dispatcher_source_contract",
    "circuit_breaker_helpers",
}
names = {case.get("case") for case in cases}
if not required.issubset(names):
    raise SystemExit(f"missing required cases: {sorted(required - names)}")

mcp_text = mcp_path.read_text(encoding="utf-8")
tools = sorted(set(re.findall(r"\| `(convir_(?:route|evidence|git)_[a-z_]+)` \|", mcp_text)))
if len(tools) != 6:
    raise SystemExit(f"MCP tool surface mismatch: {tools}")

summary_path.write_text(
    "state=PASS\n"
    f"dispatcher_cases={len(cases)}\n"
    "model_calls=0\n"
    f"mcp_tools={len(tools)}\n"
    + "mcp_tool_names=" + ",".join(tools) + "\n",
    encoding="utf-8",
)
PY

git -C "$checkout" diff --check
git -C "$checkout" diff --quiet
{
  echo "state=PASS"
  echo "candidate=$candidate"
  echo "powershell=$pwsh_version"
  echo "powershell_sha256=$observed_pwsh_sha256"
  echo "model_calls=0"
  echo "summary=$summary_path"
} > "$status_path"

cat "$summary_path"
echo "CONTROL_VALIDATION_OK candidate=$candidate run_root=$run_root"
