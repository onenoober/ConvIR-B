#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-phase4b-contract
implementation=df1308a367b14dcb36f240ed459da39f93339836
snapshot=e4ddd62ef1e6b45bec6f70b5197ef6a72de43531
catalog_sha256=d13cdfd1a13b15f6f085155dfc77630145ee539ea7bb9143d3be88db6dbebff2
terminal_record_sha256=7c896f414cb3f9d1feb07e9b8817685b3fcfea6e7225bbb887ab073e740c4530
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
test -d "$runtime_root"
work=$(mktemp -d "$runtime_root/convir-evidence-review-phase4b-real-pilot.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b-real-pilot.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
cleanup_work() {
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b-real-pilot.*) ;;
    *) printf 'refusing cleanup outside runtime root: %s\n' "$work" >&2; return 2 ;;
  esac
  rm -rf -- "$work"
  test ! -e "$work"
}
trap cleanup_work EXIT
mkdir -p "$work/tmp"

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" remote add github "$github"
git -C "$work/repo" fetch --quiet --no-tags github \
  "+refs/heads/$branch:refs/validation/candidate" \
  "+refs/heads/main:refs/remotes/github/main"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
main_tip=$(git -C "$work/repo" rev-parse refs/remotes/github/main)
test "$main_tip" = "$snapshot"
git -C "$work/repo" merge-base --is-ancestor "$implementation" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$implementation" "$candidate")
expected=$'experience_docx/CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_CONTRACT.json\nexperience_docx/tools/validate_convir_evidence_review_phase4b_real_pilot_cloud.sh'
[[ "$changed" == "$expected" ]]

predecl=$work/repo/experience_docx/CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_CONTRACT.json
tools=$work/repo/experience_docx/tools
server=$tools/convir_evidence_review_mcp.py
bash -n "$tools/validate_convir_evidence_review_phase4b_real_pilot_cloud.sh"
"$python" -m py_compile \
  "$tools/convir_evidence_cloud_inventory.py" \
  "$server"

refs_before=$work/git-refs.before
config_before=$work/git-config.before
git -C "$work/repo" for-each-ref --format='%(refname) %(objectname)' >"$refs_before"
git -C "$work/repo" config --local --null --list >"$config_before"

pilot_result=$(
  TMPDIR="$work/tmp" CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT="$runtime_root" \
  PYTHONPATH="$tools" "$python" - \
    "$server" "$work/repo" "$predecl" "$candidate" <<'PY'
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import convir_evidence_cloud_inventory as inventory

server = Path(sys.argv[1])
repo = Path(sys.argv[2])
predecl = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
candidate = sys.argv[4]

assert predecl["schema_version"] == 2
assert predecl["status"] == "AUTHORIZED_AND_FROZEN_FOR_ONE_EXECUTION"
assert predecl["implementation_commit"] == "df1308a367b14dcb36f240ed459da39f93339836"
github = predecl["github_binding"]
terminal = predecl["terminal"]
execution = predecl["execution"]
assert github == {
    "snapshot_commit": "e4ddd62ef1e6b45bec6f70b5197ef6a72de43531",
    "catalog_sha256": "d13cdfd1a13b15f6f085155dfc77630145ee539ea7bb9143d3be88db6dbebff2",
    "terminal_record_sha256": "7c896f414cb3f9d1feb07e9b8817685b3fcfea6e7225bbb887ab073e740c4530",
}
assert execution["summary_calls"] == 1
assert execution["query_calls"] == 1
assert execution["retry_after_cloud_access_boundary"] is False

# This GitHub-only gate must pass before the MCP is allowed to contact /runs.
binding = inventory.prepare_terminal_binding(
    repo,
    github["snapshot_commit"],
    github["catalog_sha256"],
    github["terminal_record_sha256"],
)
assert binding["eligible"] is True
assert binding["raw_inventory_authorized"] is True
assert binding["terminal_schema_version"] == 2
assert binding["closeout_schema_version"] == 2
assert binding["conclusion_schema_version"] == 2
assert binding["route_id"] == terminal["route_id"]
assert binding["operation_id"] == terminal["operation_id"]
assert binding["run_id"] == terminal["run_id"]
assert binding["route_commit"] == terminal["route_commit"]
assert binding["evidence_role"] == "engineering_debug"
assert binding["protected_data_permissions"] == terminal["protected_data_permissions"]
assert binding["protected_data_touched"] == terminal["protected_data_touched"]
assert binding["run_root"] == terminal["derived_run_root"]

process = subprocess.Popen(
    [sys.executable, str(server)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ.copy(),
)

def exchange(request):
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], 120)
    assert ready, f"MCP response timed out for id={request['id']}"
    line = process.stdout.readline()
    assert line
    assert len(line.encode("utf-8")) <= 32768
    response = json.loads(line)
    assert response.get("id") == request["id"]
    assert "error" not in response
    return response["result"]

initialize = exchange({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05"},
})
assert initialize["serverInfo"]["name"] == "convir-evidence-review"
assert initialize["serverInfo"]["version"] == "1.1.0"

listed = exchange({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
assert [item["name"] for item in listed["tools"]] == [
    "convir_evidence_catalog_summary",
    "convir_evidence_catalog_query",
    "convir_evidence_cloud_inventory_summary",
    "convir_evidence_cloud_inventory_query",
]

base_arguments = {
    "local_repo": str(repo),
    "snapshot_commit": github["snapshot_commit"],
    "catalog_sha256": github["catalog_sha256"],
    "terminal_record_sha256": github["terminal_record_sha256"],
}
summary_result = exchange({
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "convir_evidence_cloud_inventory_summary",
        "arguments": base_arguments,
    },
})
assert summary_result["isError"] is False
summary = summary_result["structuredContent"]
assert json.loads(summary_result["content"][0]["text"]) == summary
assert summary["ok"] is True
assert summary["state"] == "INVENTORY_READY"
assert summary["scope"] == "bound_run_root"
assert summary["root_binding_enforced"] is True
assert summary["discovery_completeness"] == "complete"
assert summary["issues"] == []
assert summary["identity"]["snapshot_commit"] == github["snapshot_commit"]
assert summary["identity"]["terminal_record_sha256"] == github["terminal_record_sha256"]
assert summary["declared_run_root"] == terminal["derived_run_root"]
assert summary["scan"]["entry_count"] > 0
assert summary["reconciliation_counts"]["MATCHED"] >= 1
inventory_sha256 = summary["inventory_sha256"]

query_arguments = {
    **base_arguments,
    "inventory_sha256": inventory_sha256,
    **execution["query"],
}
query_arguments.pop("inventory_binding")
query_result = exchange({
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "convir_evidence_cloud_inventory_query",
        "arguments": query_arguments,
    },
})
assert query_result["isError"] is False
query = query_result["structuredContent"]
assert json.loads(query_result["content"][0]["text"]) == query
assert query["ok"] is True
assert query["state"] == "INVENTORY_ENTRIES_OK"
assert query["inventory_sha256"] == inventory_sha256
assert query["complete"] is True
assert query["has_more"] is False
assert query["returned_count"] == 1
entry = query["entries"][0]
assert entry["relative_path"] == terminal["expected_formal_evidence"]
assert entry["artifact_class"] == "formal_compact_evidence"
assert entry["reconciliation_state"] == "MATCHED"
assert entry["github_sha256"] == entry["cloud_sha256"]

assert process.stdin is not None
process.stdin.close()
process.wait(timeout=15)
assert process.returncode == 0
assert process.stdout is not None
assert process.stdout.read() == ""
assert process.stderr is not None
assert process.stderr.read() == ""

print(
    "CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_OK "
    f"candidate={candidate} snapshot={github['snapshot_commit']} "
    f"terminal_record={github['terminal_record_sha256']} "
    f"inventory_sha256={inventory_sha256} matched={summary['reconciliation_counts']['MATCHED']} "
    "query_entries=1 summary_calls=1 query_calls=1 tools=4 schema_version=2 "
    "confirmation=0 canary=0 locked_test=0 dataset=0 model=0 gpu=0 writes=0"
)
PY
)

git -C "$work/repo" diff --check "$implementation" "$candidate"
git -C "$work/repo" diff --quiet
test -z "$(git -C "$work/repo" status --porcelain)"
refs_after=$work/git-refs.after
config_after=$work/git-config.after
git -C "$work/repo" for-each-ref --format='%(refname) %(objectname)' >"$refs_after"
git -C "$work/repo" config --local --null --list >"$config_after"
cmp -s "$refs_before" "$refs_after"
cmp -s "$config_before" "$config_after"

trap - EXIT
cleanup_work
printf '%s\n' "$pilot_result"
