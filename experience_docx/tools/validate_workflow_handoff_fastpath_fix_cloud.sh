#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'WORKFLOW_HANDOFF_FASTPATH_FIX_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/workflow-handoff-fastpath-fix
base=f3225fc06d7b017d24ca42eef4519e9ed416c6bf
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/workflow-handoff-fastpath-fix.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- \
  experience_docx/experiment_logs

expected=$'experience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/BRANCH_EXPERIMENT_SYNC_PROTOCOL.md\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/prepare_terminal_archive.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_prepare_terminal_archive.py\nexperience_docx/tools/validate_workflow_handoff_fastpath_fix_cloud.sh'
actual=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_prepare_terminal_archive.py"

rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$("$python" - "$stderr" <<'PY'
import re
import sys
from pathlib import Path

matches = re.findall(r"Ran ([0-9]+) tests?", Path(sys.argv[1]).read_text())
assert matches
print(matches[-1])
PY
)
test "$test_count" -ge 150

PYTHONPATH="$tools" "$python" - "$tools/convir_ops_mcp.py" <<'PY'
import json
import subprocess
import sys

server = sys.argv[1]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "workflow-handoff-acceptance", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
completed = subprocess.run(
    [sys.executable, server], input=payload, text=True, capture_output=True,
    timeout=30, check=True,
)
assert completed.stderr.strip() == "", completed.stderr
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
assert set(responses) == {1, 2}
assert responses[1]["result"]["serverInfo"]["version"] == "5.6.0"
expected = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
tools = responses[2]["result"]["tools"]
assert len(tools) == 6
assert {item["name"] for item in tools} == expected
PY

source_sha=$(sha256sum "$tools/convir_ops_mcp.py" | cut -d' ' -f1)
printf 'WORKFLOW_HANDOFF_FASTPATH_FIX_CLOUD_OK candidate=%s tests=%s source_sha256=%s version=5.6.0 tools=6 model_calls=0 gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count" "$source_sha"
