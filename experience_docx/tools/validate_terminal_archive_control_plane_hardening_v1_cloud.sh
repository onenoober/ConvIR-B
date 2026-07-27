#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/terminal-archive-control-plane-hardening-v1
base=8ed9591015e0416a97bd871f0d5e6e19f4de35bc
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/terminal-archive-control-plane-hardening.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- experience_docx/experiment_logs

tools=$work/repo/experience_docx/tools
tests=$tools/tests

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_prepare_terminal_archive.py"

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_STAGE=archive_regression\n'
PYTHONPATH="$tools:$tests" "$python" -m unittest -v test_prepare_terminal_archive

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_STAGE=full_control_plane_regression\n'
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
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 150

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_STAGE=fresh_mcp_surface\n'
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
            "clientInfo": {"name": "terminal-archive-hardening", "version": "1.0"},
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
expected_tools = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert set(responses) == {1, 2}, responses
assert responses[1]["result"]["serverInfo"]["version"] == "5.4.0"
assert {item["name"] for item in responses[2]["result"]["tools"]} == expected_tools
PY

printf 'TERMINAL_ARCHIVE_CONTROL_PLANE_HARDENING_CLOUD_OK candidate=%s tests=%s tools=6 receipt_fetch=PASS concurrent_archive=PASS gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$test_count"
