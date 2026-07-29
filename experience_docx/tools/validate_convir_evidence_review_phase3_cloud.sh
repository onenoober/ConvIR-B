#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_PHASE3_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-phase3
baseline=d6ad035c1dc02af92612fce8cfac56c18a4b3034
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
test -d "$runtime_root"
work=$(mktemp -d "$runtime_root/convir-evidence-review-phase3.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase3.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
cleanup_work() {
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase3.*) ;;
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
git -C "$work/repo" cat-file -e "$baseline^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$baseline" "$candidate")
expected=$'experience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/README.md\nexperience_docx/tools/convir_evidence_review_mcp.py\nexperience_docx/tools/tests/test_convir_evidence_review_mcp.py\nexperience_docx/tools/validate_convir_evidence_review_phase3_cloud.sh'
[[ "$changed" == "$expected" ]]

refs_before=$work/git-refs.before
config_before=$work/git-config.before
git -C "$work/repo" for-each-ref \
  --format='%(refname) %(objectname)' >"$refs_before"
git -C "$work/repo" config --local --null --list >"$config_before"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
server=$tools/convir_evidence_review_mcp.py
"$python" -m py_compile \
  "$server" \
  "$tools/convir_evidence_catalog.py" \
  "$tests/test_convir_evidence_review_mcp.py"

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
set +e
TMPDIR="$work/tmp" PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 120 "$stdout" >&2 || true
  tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 283

TMPDIR="$work/tmp" CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT="$runtime_root" \
PYTHONPATH="$tools" "$python" - "$server" "$work/repo" "$main_tip" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

server = Path(sys.argv[1])
repo = Path(sys.argv[2])
main_tip = sys.argv[3]
long_id = "\x01" * 20
oversized_id = "z" * 32768
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "convir_evidence_catalog_summary",
            "arguments": {"local_repo": str(repo)},
        },
    },
    {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {
            "name": "convir_evidence_catalog_query",
            "arguments": {
                "local_repo": str(repo), "snapshot_commit": main_tip,
                "coverage": "indexed", "terms": ["haze4k"], "limit": 5,
            },
        },
    },
    {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {
            "name": "convir_evidence_catalog_query",
            "arguments": {
                "local_repo": str(repo), "snapshot_commit": main_tip,
                "coverage": "unindexed",
                "terms": ["run_v2f_f4b_tail_rescue_matrix.sh"], "limit": 1,
            },
        },
    },
    {
        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {
            "name": "convir_evidence_catalog_query",
            "arguments": {
                "local_repo": str(repo), "snapshot_commit": main_tip,
                "terms": ["term"] * 9,
            },
        },
    },
    {
        "jsonrpc": "2.0", "id": long_id, "method": "tools/call",
        "params": {
            "name": "convir_evidence_catalog_query",
            "arguments": {
                "local_repo": str(repo), "snapshot_commit": main_tip,
                "coverage": "all", "limit": 100,
            },
        },
    },
    {"jsonrpc": "2.0", "id": oversized_id, "method": "ping", "params": {}},
]
completed = subprocess.run(
    [sys.executable, str(server)],
    input="".join(json.dumps(item) + "\n" for item in requests),
    text=True,
    capture_output=True,
    check=True,
    timeout=60,
    env=os.environ.copy(),
)
assert completed.stderr == ""
lines = completed.stdout.splitlines()
assert len(lines) == len(requests)
assert all(len(line.encode()) + 1 <= 32768 for line in lines)
responses = [json.loads(line) for line in lines]
assert [item["id"] for item in responses] == [1, 2, 3, 4, 5, 6, long_id, None]
assert all("error" not in item for item in responses[:7])
assert "error" in responses[7]

info = responses[0]["result"]["serverInfo"]
assert info["name"] == "convir-evidence-review"
assert info["version"] == "1.0.0"
assert info["sourceSha256"] == hashlib.sha256(server.read_bytes()).hexdigest()
assert info["catalogSourceSha256"] == hashlib.sha256(
    (server.parent / "convir_evidence_catalog.py").read_bytes()
).hexdigest()
assert info["transportSourceSha256"] == hashlib.sha256(
    (server.parent / "convirctl.py").read_bytes()
).hexdigest()

listed = responses[1]["result"]["tools"]
assert [item["name"] for item in listed] == [
    "convir_evidence_catalog_summary", "convir_evidence_catalog_query",
]
assert all("outputSchema" in item for item in listed)
summary = responses[2]["result"]
indexed = responses[3]["result"]
loose = responses[4]["result"]
bounded_filter = responses[5]["result"]
bulk = responses[6]["result"]
for result in (summary, indexed, loose, bounded_filter, bulk):
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
assert summary["isError"] is False
summary_value = summary["structuredContent"]
header = summary_value["header"]
assert header["snapshot_commit"] == main_tip
assert header["scientific_completeness"] == "not_assessed"
assert header["excluded_sources"] == ["route_branches", "cloud_runtime"]
assert summary_value["trusted_remote"] == "github"
assert summary_value["trusted_remote_url"] == "git@github.com:onenoober/ConvIR-B.git"
assert summary_value["trusted_ref"] == "refs/remotes/github/main"
assert summary_value["trusted_ref_tip"] == main_tip
assert summary_value["ref_freshness"] == "not_assessed"
assert summary_value["git_mutations_performed"] is False
assert header["terminal_index"]["record_count"] >= 55
assert header["terminal_index"]["route_count"] >= 54
tree = header["experiment_log_tree"]
assert tree["catalog_entry_count"] >= 232
assert tree["directory_count"] >= 231
assert tree["indexed_directory_count"] >= 54
assert tree["catalog_entry_count"] == tree["directory_count"] + tree["loose_file_count"]
assert tree["directory_count"] == (
    tree["indexed_directory_count"] + tree["unindexed_directory_count"]
)
assert indexed["isError"] is False
assert indexed["structuredContent"]["total_count"] > 5
assert indexed["structuredContent"]["returned_count"] <= 5
assert loose["isError"] is False
assert loose["structuredContent"]["total_count"] == 1
loose_entry = loose["structuredContent"]["entries"][0]
assert loose_entry["record_kind"] == "loose_file"
assert loose_entry["terminal_assessment"] == "NOT_ASSESSED"
assert bounded_filter["isError"] is True
assert bounded_filter["structuredContent"]["state"] == "ARGUMENTS_INVALID"
assert bulk["isError"] is False
assert 0 < bulk["structuredContent"]["returned_count"] <= 100
PY

git -C "$work/repo" diff --check "$baseline" "$candidate"
git -C "$work/repo" diff --quiet
test -z "$(git -C "$work/repo" status --porcelain)"
refs_after=$work/git-refs.after
config_after=$work/git-config.after
git -C "$work/repo" for-each-ref \
  --format='%(refname) %(objectname)' >"$refs_after"
git -C "$work/repo" config --local --null --list >"$config_after"
cmp -s "$refs_before" "$refs_after"
cmp -s "$config_before" "$config_after"
trap - EXIT
cleanup_work
printf 'CONVIR_EVIDENCE_REVIEW_PHASE3_CLOUD_OK candidate=%s github_main=%s tests=%s tools=2 catalog_entries_at_least=232 git_mutations=0 cloud_runtime_access=0 model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$main_tip" "$test_count"
