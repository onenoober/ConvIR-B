#!/bin/bash
set -euo pipefail

BRANCH=codex/p0-p1-research-governance-fastpath-20260721
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
SEED=$BASE/repos/ConvIR-B-official-arch-anchor

REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH")
read -r REMOTE_COMMIT REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_REF" = "refs/heads/$BRANCH"
[[ "$REMOTE_COMMIT" =~ ^[0-9a-f]{40}$ ]]

RUN_ROOT=$BASE/runs/p0-p1-governance-validation/$REMOTE_COMMIT/final-acceptance
REPO=$BASE/repos/p0-p1-governance-final-$REMOTE_COMMIT
STATUS=$RUN_ROOT/status.txt
LOG=$RUN_ROOT/unittest.log
ACCEPTANCE=$RUN_ROOT/acceptance.json
mkdir -p "$RUN_ROOT"
if test -e "$STATUS"; then
  printf 'existing validation status: %s\n' "$(tr '\n' ' ' < "$STATUS")"
  exit 73
fi
printf 'state=PREPARING\nbranch=%s\ncommit=%s\n' "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"

on_exit() {
  code=$?
  if test "$code" -ne 0; then
    printf 'state=FAILED_ENGINEERING\nexit_code=%s\n' "$code" >> "$STATUS"
  fi
}
trap on_exit EXIT

test -d "$SEED/.git" || test -f "$SEED/HEAD"
test ! -e "$REPO"
/usr/bin/git clone --quiet --no-checkout --reference-if-able "$SEED" "$REMOTE_URL" "$REPO"
/usr/bin/git -C "$REPO" remote rename origin github
/usr/bin/git -C "$REPO" fetch --quiet --no-tags github \
  "+refs/heads/$BRANCH:refs/validation/candidate" \
  "+refs/heads/main:refs/remotes/github/main"
test "$(/usr/bin/git -C "$REPO" rev-parse refs/validation/candidate)" = "$REMOTE_COMMIT"
/usr/bin/git -C "$REPO" checkout --quiet --detach "$REMOTE_COMMIT"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"
printf 'state=RUNNING\nbranch=%s\ncommit=%s\n' "$BRANCH" "$REMOTE_COMMIT" > "$STATUS"

TOOLS=$REPO/experience_docx/tools
TESTS=$TOOLS/tests
export PYTHONPATH=$TOOLS:$TESTS

"$PYTHON" -m py_compile \
  "$TOOLS/convir_ops_mcp.py" \
  "$TOOLS/research_program_contract.py" \
  "$TOOLS/experiment_spec_compiler.py" \
  "$TOOLS/research_workspace.py" \
  "$TOOLS/policy_snapshot.py" \
  "$TOOLS/capability_registry.py" \
  "$TOOLS/validate_route_ready.py"

set +e
"$PYTHON" -m unittest discover -s "$TESTS" -p 'test_*.py' > "$LOG" 2>&1
TEST_RC=$?
set -e
if test "$TEST_RC" -ne 0; then
  tail -n 240 "$LOG" >&2 || true
  exit "$TEST_RC"
fi
TEST_COUNT=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$LOG" | tail -n 1)
[[ "$TEST_COUNT" =~ ^[0-9]+$ ]]
test "$TEST_COUNT" -ge 140
grep -Fq 'OK' "$LOG"

"$PYTHON" - "$REPO" "$REMOTE_COMMIT" "$ACCEPTANCE" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

import capability_registry as registry
import convir_ops_mcp as ops
import policy_snapshot

repo = Path(sys.argv[1])
candidate = sys.argv[2]
output = Path(sys.argv[3])
snapshot_path = repo / policy_snapshot.SNAPSHOT_RELPATH
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
policy_snapshot.verify_snapshot(
    snapshot, read_bytes=lambda relpath: (repo / relpath).read_bytes(),
)
rules_commit = snapshot["rules_commit"]
subprocess.run(
    ["/usr/bin/git", "-C", str(repo), "merge-base", "--is-ancestor", rules_commit, candidate],
    check=True,
)
for relpath in policy_snapshot.POLICY_SOURCES:
    subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "diff", "--quiet", rules_commit, candidate, "--", relpath],
        check=True,
    )
snapshot_bytes = snapshot_path.stat().st_size
assert 1024 <= snapshot_bytes <= 16384, snapshot_bytes

records = registry.load_records(
    (repo / registry.REGISTRY_RELPATH).read_text(encoding="utf-8").splitlines(),
    evidence_exists=lambda relpath: (repo / relpath).is_file(),
    read_evidence=lambda relpath: (repo / relpath).read_bytes(),
)
assert records == [], records

server = repo / "experience_docx/tools/convir_ops_mcp.py"
requests = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "convir_git_status",
            "arguments": {
                "local_repo": str(repo), "route_id": "p0_p1_governance",
                "detail": "summary",
            },
        },
    },
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
environment = os.environ.copy()
environment["CONVIR_OPS_LOCAL_WORKSPACE_ROOT"] = str(repo.parent)
completed = subprocess.run(
    [sys.executable, str(server)], input=payload, text=True, capture_output=True,
    env=environment, timeout=30, check=True,
)
assert not completed.stderr.strip(), completed.stderr[:2000]
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
assert set(responses) == {1, 2, 3}, responses
assert all("error" not in responses[index] for index in responses), responses
initialize = responses[1]["result"]
tools = responses[2]["result"]["tools"]
call = responses[3]["result"]
assert initialize["serverInfo"]["version"] == "5.1.0"
expected_tools = {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert len(tools) == 6 and {item["name"] for item in tools} == expected_tools
finish = next(item for item in tools if item["name"] == "convir_route_finish")
assert finish["inputSchema"]["properties"]["engineering_failure_resolution"]["enum"] == ["repair", "archive", "discard"]
assert sorted(ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS) == [4, 5, 6]
assert call["isError"] is False, call
summary = call["content"][0]["text"]
assert len(summary.encode()) <= 2048
compact = ops.typed_result(
    True, "READY", plan_token="a" * 64, receipt="b" * 64,
)
compact_text = compact["content"][0]["text"]
assert "plan_token" in compact_text and "receipt" in compact_text
assert len(compact_text.encode()) <= 512

result = {
    "schema_version": 1,
    "status": "COMPLETED_GATE_PASS",
    "decision": "P0_P1_GOVERNANCE_ADOPTION_CANDIDATE_PASS",
    "candidate_commit": candidate,
    "rules_commit": rules_commit,
    "policy_snapshot_sha256": __import__("hashlib").sha256(snapshot_path.read_bytes()).hexdigest(),
    "policy_snapshot_bytes": snapshot_bytes,
    "server_version": initialize["serverInfo"]["version"],
    "server_source_sha256": initialize["serverInfo"]["sourceSha256"],
    "tool_count": len(tools),
    "manifest_schemas": sorted(ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS),
    "git_status_summary_bytes": len(summary.encode()),
    "token_receipt_summary_bytes": len(compact_text.encode()),
    "capability_registry_records": len(records),
    "model_calls": 0, "gpu_access": 0, "dataset_access": 0,
    "checkpoint_access": 0, "confirmation_access": 0,
    "canary_access": 0, "locked_test_access": 0,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY

printf 'state=COMPLETED_GATE_PASS\nbranch=%s\ncommit=%s\ntests=%s\nmarker=P0_P1_GOVERNANCE_FINAL_CLOUD_OK\n' \
  "$BRANCH" "$REMOTE_COMMIT" "$TEST_COUNT" > "$STATUS"
cat "$ACCEPTANCE"
echo P0_P1_GOVERNANCE_FINAL_CLOUD_OK
