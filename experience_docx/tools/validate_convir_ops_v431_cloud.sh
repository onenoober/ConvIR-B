#!/usr/bin/env bash
set -euo pipefail

BRANCH=codex/convir-ops-v4-3-1-migrated-repair-reopen-20260717
GITHUB=git@github.com:onenoober/ConvIR-B.git
SEED=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
PYTHON=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
WORK=$(mktemp -d /tmp/convir-ops-v431-acceptance.XXXXXX)
trap 'rm -rf -- "$WORK"' EXIT

git clone --quiet --shared --no-checkout "$SEED" "$WORK/repo"
git -C "$WORK/repo" fetch --quiet "$GITHUB" "refs/heads/$BRANCH"
git -C "$WORK/repo" checkout --quiet --detach FETCH_HEAD
test -z "$(git -C "$WORK/repo" status --porcelain)"

cd "$WORK/repo"
"$PYTHON" -m unittest discover \
  -s experience_docx/tools/tests \
  -p 'test_*.py' \
  -v

"$PYTHON" - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

server = Path("experience_docx/tools/convir_ops_mcp.py")
requests = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
completed = subprocess.run(
    ["/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python", str(server)],
    input="".join(json.dumps(item) + "\n" for item in requests),
    text=True,
    capture_output=True,
    check=True,
    timeout=30,
)
responses = [json.loads(line) for line in completed.stdout.splitlines()]
server_info = responses[0]["result"]["serverInfo"]
assert server_info["name"] == "convir-ops"
assert server_info["version"] == "4.3.1"
assert server_info["sourceSha256"] == hashlib.sha256(server.read_bytes()).hexdigest()
tools = responses[1]["result"]["tools"]
assert len(tools) == 6
finish = next(item for item in tools if item["name"] == "convir_route_finish")
assert finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"] == ["repair", "archive"]
print(json.dumps({
    "server_version": "4.3.1",
    "tool_count": len(tools),
    "finish_resolution_enum": ["repair", "archive"],
    "source_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
}, sort_keys=True))
PY

printf '%s\n' CONVIR_OPS_V431_CLOUD_ACCEPTANCE_OK
