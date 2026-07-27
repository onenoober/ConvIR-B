#!/bin/bash
set -euo pipefail

python_bin=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
seed_repo=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
branch=codex/general-science-control-plane-v1
temporary_root=$(/usr/bin/mktemp -d /sda/home/wangyuxin/ConvIR-B/tmp/control-plane-acceptance.XXXXXX)
repository=${temporary_root}/repo

cleanup() {
  case "${temporary_root}" in
    /sda/home/wangyuxin/ConvIR-B/tmp/control-plane-acceptance.*)
      /bin/rm -rf -- "${temporary_root}"
      ;;
    *)
      echo "refusing unsafe temporary cleanup: ${temporary_root}" >&2
      ;;
  esac
}
trap cleanup EXIT

/usr/bin/git clone --quiet --shared --no-checkout "${seed_repo}" "${repository}"
/usr/bin/git -C "${repository}" remote remove github 2>/dev/null || true
/usr/bin/git -C "${repository}" remote add github git@github.com:onenoober/ConvIR-B.git
/usr/bin/git -C "${repository}" fetch --quiet github \
  "refs/heads/${branch}:refs/remotes/github/${branch}" \
  "refs/heads/main:refs/remotes/github/main"
/usr/bin/git -C "${repository}" checkout --quiet --detach "refs/remotes/github/${branch}"

head=$(/usr/bin/git -C "${repository}" rev-parse HEAD)
if test -n "$(/usr/bin/git -C "${repository}" status --porcelain)"; then
  echo "acceptance checkout is dirty" >&2
  exit 2
fi

cd "${repository}"
export PYTHONPATH="${repository}/experience_docx/tools:${repository}/experience_docx/tools/tests"
"${python_bin}" -m unittest discover \
  -s experience_docx/tools/tests \
  -p 'test_*.py' \
  -v

CONVIR_OPS_LOCAL_WORKSPACE_ROOT="${temporary_root}" "${python_bin}" - \
  "${repository}" <<'PY'
import json
import hashlib
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
server = repo / "experience_docx/tools/convir_ops_mcp.py"
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "control-plane-acceptance", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "convir_git_status",
            "arguments": {
                "local_repo": str(repo), "route_id": "control-plane-acceptance",
                "detail": "summary",
            },
        },
    },
]
payload = "".join(
    json.dumps(item, separators=(",", ":")) + "\n" for item in requests
)
completed = subprocess.run(
    [sys.executable, str(server)], input=payload, text=True,
    capture_output=True, timeout=30, check=True,
)
if completed.stderr.strip():
    raise AssertionError(f"unexpected MCP stderr: {completed.stderr[:1000]}")
responses = {}
for line in completed.stdout.splitlines():
    response = json.loads(line)
    responses[response["id"]] = response
assert set(responses) == {1, 2, 3}, responses
assert all("error" not in responses[index] for index in responses), responses
initialize = responses[1]["result"]
tools = responses[2]["result"]["tools"]
status = responses[3]["result"]
expected_tools = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert initialize["serverInfo"]["version"] == "5.4.0", initialize
assert initialize["protocolVersion"] == "2024-11-05", initialize
expected_source_sha = hashlib.sha256(server.read_bytes()).hexdigest()
assert initialize["serverInfo"]["sourceSha256"] == expected_source_sha, initialize
assert len(tools) == 6 and {item["name"] for item in tools} == expected_tools, tools
assert status["isError"] is False, status
print("GENERAL_SCIENCE_CONTROL_PLANE_FRESH_MCP_OK version=5.4.0 tools=6")
PY

echo "GENERAL_SCIENCE_CONTROL_PLANE_ACCEPTANCE_OK commit=${head}"
