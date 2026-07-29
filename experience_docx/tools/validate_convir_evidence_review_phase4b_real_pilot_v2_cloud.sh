#!/usr/bin/env bash
set -euo pipefail

branch=codex/convir-evidence-review-phase4b-contract
base=9d660a379da858177170e949f3cf074ee3c13f9c
implementation=df1308a367b14dcb36f240ed459da39f93339836
snapshot=e4ddd62ef1e6b45bec6f70b5197ef6a72de43531
catalog_sha256=d13cdfd1a13b15f6f085155dfc77630145ee539ea7bb9143d3be88db6dbebff2
terminal_record_sha256=7c896f414cb3f9d1feb07e9b8817685b3fcfea6e7225bbb887ab073e740c4530
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
receipt_dir=$runtime_root/convir-evidence-review/receipts
receipt_path=$receipt_dir/convir-evidence-review-phase4b-real-pilot-v2-repair1.json

work=
stage=bootstrap
candidate=
script_sha256=
contract_sha256=
pilot_result_json=
temporary_workspace_removed=false

write_receipt() {
  local state=$1
  local accepted=$2
  local exit_code=$3
  local failure_stage=$4
  RECEIPT_STATE="$state" \
  RECEIPT_ACCEPTED="$accepted" \
  RECEIPT_EXIT_CODE="$exit_code" \
  RECEIPT_FAILURE_STAGE="$failure_stage" \
  RECEIPT_CANDIDATE="$candidate" \
  RECEIPT_SCRIPT_SHA256="$script_sha256" \
  RECEIPT_CONTRACT_SHA256="$contract_sha256" \
  RECEIPT_RESULT_JSON="$pilot_result_json" \
  RECEIPT_WORK_REMOVED="$temporary_workspace_removed" \
  "$python" - "$receipt_dir" "$receipt_path" <<'PY'
import json
import os
import sys
from pathlib import Path

receipt_dir = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
receipt_dir.mkdir(parents=True, exist_ok=True)
if receipt_path.exists() or receipt_path.is_symlink():
    raise FileExistsError(receipt_path)
result_text = os.environ.get("RECEIPT_RESULT_JSON", "")
outcome = json.loads(result_text) if result_text else {}
if outcome:
    outcome["temporary_workspace_removed"] = os.environ["RECEIPT_WORK_REMOVED"] == "true"
payload = {
    "schema_version": 2,
    "receipt_id": "convir-evidence-review-phase4b-real-pilot-v2-repair1-receipt",
    "pilot_id": "convir-evidence-review-phase4b-real-pilot-v2",
    "state": os.environ["RECEIPT_STATE"],
    "accepted": os.environ["RECEIPT_ACCEPTED"] == "true",
    "exit_code": int(os.environ["RECEIPT_EXIT_CODE"]),
    "failure_stage": os.environ["RECEIPT_FAILURE_STAGE"] or None,
    "candidate_commit": os.environ["RECEIPT_CANDIDATE"] or None,
    "implementation_commit": "df1308a367b14dcb36f240ed459da39f93339836",
    "github_binding": {
        "snapshot_commit": "e4ddd62ef1e6b45bec6f70b5197ef6a72de43531",
        "catalog_sha256": "d13cdfd1a13b15f6f085155dfc77630145ee539ea7bb9143d3be88db6dbebff2",
        "terminal_record_sha256": "7c896f414cb3f9d1feb07e9b8817685b3fcfea6e7225bbb887ab073e740c4530",
    },
    "source_identity": {
        "validation_script_sha256": os.environ["RECEIPT_SCRIPT_SHA256"] or None,
        "pilot_contract_sha256": os.environ["RECEIPT_CONTRACT_SHA256"] or None,
    },
    "outcome": outcome,
    "access_observed": {
        "confirmation": 0,
        "canary": 0,
        "locked_test": 0,
        "dataset": 0,
        "model": 0,
        "checkpoint": 0,
        "gpu": 0,
        "run_root_writes": 0,
        "git_mutations": 0,
    },
}
data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
assert len(data) <= 16384
tmp = receipt_dir / ("." + receipt_path.name + "." + str(os.getpid()) + ".tmp")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(tmp, flags, 0o444)
try:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    os.fsync(fd)
finally:
    os.close(fd)
try:
    os.link(tmp, receipt_path)
finally:
    tmp.unlink(missing_ok=True)
dir_fd = os.open(receipt_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
PY
}

cleanup_work() {
  if [[ -z "$work" ]]; then
    temporary_workspace_removed=true
    return 0
  fi
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b-real-pilot-v2.*) ;;
    *) printf 'refusing cleanup outside runtime root: %s\n' "$work" >&2; return 2 ;;
  esac
  rm -rf -- "$work"
  test ! -e "$work"
  work=
  temporary_workspace_removed=true
}

on_exit() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if [[ -n "$work" ]]; then
    cleanup_work || true
  fi
  if [[ "$rc" -ne 0 && ! -e "$receipt_path" ]]; then
    write_receipt "PHASE4B_REAL_PILOT_FAILED" false "$rc" "$stage" || true
  fi
  if [[ "$rc" -ne 0 ]]; then
    printf 'CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V2_FAILED stage=%s rc=%s\n' "$stage" "$rc" >&2
  fi
  exit "$rc"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

stage=receipt_preflight
test -d "$runtime_root"
mkdir -p "$receipt_dir"
test ! -e "$receipt_path"

stage=github_checkout
work=$(mktemp -d "$runtime_root/convir-evidence-review-phase4b-real-pilot-v2.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b-real-pilot-v2.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
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
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
expected=$'experience_docx/CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V1_OUTCOME.json\nexperience_docx/CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V2_CONTRACT.json\nexperience_docx/tools/inspect_convir_evidence_review_phase4b_real_pilot_v2_receipt_cloud.sh\nexperience_docx/tools/validate_convir_evidence_review_phase4b_real_pilot_v2_cloud.sh'
[[ "$changed" == "$expected" ]]

predecl=$work/repo/experience_docx/CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V2_CONTRACT.json
tools=$work/repo/experience_docx/tools
server=$tools/convir_evidence_review_mcp.py
validation_script=$tools/validate_convir_evidence_review_phase4b_real_pilot_v2_cloud.sh
bash -n "$validation_script"
bash -n "$tools/inspect_convir_evidence_review_phase4b_real_pilot_v2_receipt_cloud.sh"
"$python" -m py_compile \
  "$tools/convir_evidence_cloud_inventory.py" \
  "$server"
script_sha256=$(sha256sum "$validation_script" | cut -d' ' -f1)
contract_sha256=$(sha256sum "$predecl" | cut -d' ' -f1)

refs_before=$work/git-refs.before
config_before=$work/git-config.before
git -C "$work/repo" for-each-ref --format='%(refname) %(objectname)' >"$refs_before"
git -C "$work/repo" config --local --null --list >"$config_before"

stage=github_binding_gate
result_path=$work/pilot-result.json
TMPDIR="$work/tmp" CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT="$runtime_root" \
PYTHONPATH="$tools" "$python" - \
  "$server" "$work/repo" "$predecl" "$candidate" "$result_path" <<'PY'
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
result_path = Path(sys.argv[5])

assert predecl["schema_version"] == 2
assert predecl["status"] == "AUTHORIZED_FOR_ONE_SAME_CONTRACT_ENGINEERING_REPAIR"
assert predecl["implementation_commit"] == "df1308a367b14dcb36f240ed459da39f93339836"
assert predecl["source_base_commit"] == "9d660a379da858177170e949f3cf074ee3c13f9c"
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
assert execution["repair_cycle"] == 1
assert execution["repair_scope"] == "VALIDATION_TRANSPORT_ONLY"
assert execution["repair_transport"] == "CLOUD_LOCAL_SAME_COMMIT_REMOTE_WORKER"
assert execution["production_transport_changed"] is False

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
assert binding["conclusion_schema_version"] is None
assert binding["conclusion_schema_state"] == "LEGACY_UNVERSIONED"
assert binding["route_id"] == terminal["route_id"]
assert binding["operation_id"] == terminal["operation_id"]
assert binding["run_id"] == terminal["run_id"]
assert binding["route_commit"] == terminal["route_commit"]
assert binding["evidence_role"] == "engineering_debug"
assert binding["protected_data_permissions"] == terminal["protected_data_permissions"]
assert binding["protected_data_touched"] == terminal["protected_data_touched"]
assert binding["run_root"] == terminal["derived_run_root"]

launcher = """
import convir_evidence_cloud_inventory as inventory
import convir_evidence_review_mcp as review

assert review.REMOTE_HOST == "convir-4090"
assert review.REMOTE_PYTHON == "/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
assert review.SSH == "/usr/bin/ssh"

def cloud_local_same_commit_worker(request):
    return inventory.remote_worker(request)

review._run_fixed_remote = cloud_local_same_commit_worker
review.main()
"""
process = subprocess.Popen(
    [sys.executable, "-c", launcher],
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

try:
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
    assert summary_result["isError"] is False, json.dumps(summary_result, sort_keys=True)
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
    assert query_result["isError"] is False, json.dumps(query_result, sort_keys=True)
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
finally:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=15)

assert process.returncode == 0
assert process.stdout is not None
assert process.stdout.read() == ""
assert process.stderr is not None
assert process.stderr.read() == ""

result = {
    "schema_version": 2,
    "candidate_commit": candidate,
    "inventory_sha256": inventory_sha256,
    "matched_count": summary["reconciliation_counts"]["MATCHED"],
    "query_entries": 1,
    "summary_calls": 1,
    "query_calls": 1,
    "repair_cycle": 1,
    "repair_transport": "CLOUD_LOCAL_SAME_COMMIT_REMOTE_WORKER",
    "tools": 4,
    "terminal_schema_version": 2,
    "closeout_schema_version": 2,
    "conclusion_schema_state": "LEGACY_UNVERSIONED",
}
result_path.write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

stage=post_access_integrity
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet
test -z "$(git -C "$work/repo" status --porcelain)"
refs_after=$work/git-refs.after
config_after=$work/git-config.after
git -C "$work/repo" for-each-ref --format='%(refname) %(objectname)' >"$refs_after"
git -C "$work/repo" config --local --null --list >"$config_after"
cmp -s "$refs_before" "$refs_after"
cmp -s "$config_before" "$config_after"
pilot_result_json=$(tr -d '\n' <"$result_path")

stage=temporary_cleanup
cleanup_work

stage=receipt_publish
write_receipt "PHASE4B_REAL_PILOT_PASS" true 0 ""
receipt_sha256=$(sha256sum "$receipt_path" | cut -d' ' -f1)
inventory_sha256=$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["inventory_sha256"])' "$pilot_result_json")

stage=complete
printf '%s\n' \
  "CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_V2_OK candidate=$candidate snapshot=$snapshot terminal_record=$terminal_record_sha256 inventory_sha256=$inventory_sha256 receipt_sha256=$receipt_sha256 summary_calls=1 query_calls=1 tools=4 schema_version=2 confirmation=0 canary=0 locked_test=0 dataset=0 model=0 checkpoint=0 gpu=0 run_root_writes=0 git_mutations=0"
