#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_MAIN_ADOPTION_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-loop-v2
baseline=479072498570665bdad4c2ae376aa397aea6880c
rules_commit=d447f9a5afe8ebde08340a4077bfd73228ff24bc
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
test -d "$runtime_root"
work=$(mktemp -d "$runtime_root/convir-evidence-review-main-adoption.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-main-adoption.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
cleanup_work() {
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-main-adoption.*) ;;
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
test "$main_tip" = "$baseline"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
git -C "$work/repo" merge-base --is-ancestor "$rules_commit" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$baseline" "$candidate")
expected=$'experience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_evidence_review_mcp.py\nexperience_docx/tools/policy_snapshot.py\nexperience_docx/tools/tests/test_convir_evidence_review_mcp.py\nexperience_docx/tools/validate_convir_evidence_review_main_adoption_cloud.sh'
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
  "$tools/policy_snapshot.py" \
  "$tools/convir_evidence_catalog.py" \
  "$tools/convir_evidence_cloud_inventory.py" \
  "$server" \
  "$tests/test_convir_evidence_catalog.py" \
  "$tests/test_convir_evidence_cloud_inventory.py" \
  "$tests/test_convir_evidence_review_mcp.py"
bash -n "$tools/validate_convir_evidence_review_main_adoption_cloud.sh"
"$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

focused_stdout=$work/focused.stdout
focused_stderr=$work/focused.stderr
if TMPDIR="$work/tmp" PYTHONPATH="$tools:$tests" "$python" -m unittest \
    test_convir_evidence_catalog \
    test_convir_evidence_cloud_inventory \
    test_convir_evidence_review_mcp \
    >"$focused_stdout" 2>"$focused_stderr"; then
  focused_rc=0
else
  focused_rc=$?
fi
if [[ $focused_rc -ne 0 ]]; then
  tail -n 160 "$focused_stdout" >&2 || true
  tail -n 160 "$focused_stderr" >&2 || true
  exit "$focused_rc"
fi
focused_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$focused_stderr" | tail -n 1)
test "$focused_count" -ge 20

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
if TMPDIR="$work/tmp" PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
    -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"; then
  rc=0
else
  rc=$?
fi
if [[ $rc -ne 0 ]]; then
  tail -n 160 "$stdout" >&2 || true
  tail -n 160 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
test "$test_count" -ge 312

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
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "convir_evidence_completeness_receipt",
            "arguments": {"local_repo": str(repo)},
        },
    },
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
raw_lines = completed.stdout.splitlines(keepends=True)
assert len(raw_lines) == 3
assert all(len(line.encode("utf-8")) <= 32768 for line in raw_lines)
responses = [json.loads(line) for line in raw_lines]
assert all("error" not in response for response in responses)
info = responses[0]["result"]["serverInfo"]
assert info["name"] == "convir-evidence-review"
assert info["version"] == "1.4.0"
assert info["sourceSha256"] == hashlib.sha256(server.read_bytes()).hexdigest()
assert [item["name"] for item in responses[1]["result"]["tools"]] == [
    "convir_evidence_catalog_summary",
    "convir_evidence_completeness_receipt",
    "convir_evidence_catalog_query",
    "convir_evidence_bundle",
    "convir_evidence_cloud_inventory_summary",
    "convir_evidence_cloud_inventory_query",
    "convir_evidence_cloud_text_read",
]
result = responses[2]["result"]
assert result["isError"] is False
value = result["structuredContent"]
assert json.loads(result["content"][0]["text"]) == value
assert value["schema_version"] == 2
assert value["snapshot_commit"] == main_tip
assert value["review_completeness"] == "incomplete"
assert value["scientific_completeness"] == "not_assessed"
assert value["entry_partition"]["catalog_entries"] == 232
assert value["entry_partition"]["unindexed_entries"] == 178
assert value["terminal_partition"]["terminal_records"] == 55
assert value["terminal_partition"]["routes"] == 54
assert value["unresolved_counts"]["ambiguous_legacy_routes"] == 1
assert value["git_mutations_performed"] is False
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
printf 'CONVIR_EVIDENCE_REVIEW_MAIN_ADOPTION_CLOUD_OK candidate=%s github_main=%s rules_commit=%s tests=%s focused_tests=%s tools=7 cloud_access=0 model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$main_tip" "$rules_commit" "$test_count" "$focused_count"
