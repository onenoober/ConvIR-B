#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_PHASE4A_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-phase4a
baseline=915e93ef24eb6121614e7d4e93e084b874e7ee66
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
test -d "$runtime_root"
work=$(mktemp -d "$runtime_root/convir-evidence-review-phase4a.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4a.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
cleanup_work() {
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4a.*) ;;
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
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$baseline" "$candidate")
expected=$'experience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/tools/convir_evidence_catalog.py\nexperience_docx/tools/convir_evidence_cloud_inventory.py\nexperience_docx/tools/tests/test_convir_evidence_cloud_inventory.py\nexperience_docx/tools/validate_convir_evidence_review_phase4a_cloud.sh'
[[ "$changed" == "$expected" ]]

refs_before=$work/git-refs.before
config_before=$work/git-config.before
git -C "$work/repo" for-each-ref \
  --format='%(refname) %(objectname)' >"$refs_before"
git -C "$work/repo" config --local --null --list >"$config_before"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
server=$tools/convir_evidence_review_mcp.py
inventory=$tools/convir_evidence_cloud_inventory.py
inventory_test=$tests/test_convir_evidence_cloud_inventory.py
"$python" -m py_compile \
  "$tools/convir_evidence_catalog.py" \
  "$inventory" \
  "$inventory_test"
bash -n "$tools/validate_convir_evidence_review_phase4a_cloud.sh"

focused_stdout=$work/focused.stdout
focused_stderr=$work/focused.stderr
if TMPDIR="$work/tmp" PYTHONPATH="$tools:$tests" "$python" -m unittest \
    test_convir_evidence_cloud_inventory \
    >"$focused_stdout" 2>"$focused_stderr"; then
  focused_rc=0
else
  focused_rc=$?
fi
if [[ $focused_rc -ne 0 ]]; then
  tail -n 120 "$focused_stdout" >&2 || true
  tail -n 120 "$focused_stderr" >&2 || true
  exit "$focused_rc"
fi
focused_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$focused_stderr" | tail -n 1)
test "$focused_count" = 18

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
if TMPDIR="$work/tmp" PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
    -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"; then
  rc=0
else
  rc=$?
fi
if [[ $rc -ne 0 ]]; then
  tail -n 120 "$stdout" >&2 || true
  tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 301

TMPDIR="$work/tmp" CONVIR_EVIDENCE_LOCAL_WORKSPACE_ROOT="$runtime_root" \
PYTHONPATH="$tools" "$python" - "$server" "$work/repo" "$main_tip" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import convir_evidence_catalog as catalog
import convir_evidence_cloud_inventory as inventory

server = Path(sys.argv[1])
repo = Path(sys.argv[2])
main_tip = sys.argv[3]
loaded = catalog.load_catalog(repo, main_tip)
assert loaded["catalog_sha256"] == "30994fb55e72b86f12e54b33562d5d5ff91ee70313597510955cdd4a578adf0f"
assert loaded["collection_sha256"] == "181d09fe2c4e6080192be99a3cbc286ac44db37aee327938461fd42ec353f92a"
_, records, _ = catalog.load_terminal_records(repo, main_tip)
schema2 = [record for record in records if record["schema_version"] == 2]
assert len(schema2) == 8
conclusion_schema_states = []
for record in schema2:
    binding = inventory.prepare_terminal_binding(
        repo, main_tip, loaded["catalog_sha256"], record["record_sha256"]
    )
    assert binding["eligible"] is True
    assert binding["state"] == "TERMINAL_BINDING_VERIFIED"
    assert binding["run_id"] == binding["output_id"]
    assert binding["scientific_completeness"] == "not_assessed"
    assert binding["terminal_schema_version"] == 2
    assert binding["closeout_schema_version"] == 2
    conclusion_schema_states.append(binding["conclusion_schema_state"])
    assert binding["run_root"] == (
        f"{inventory.REMOTE_RUNS}/{binding['route_id']}/{binding['output_id']}"
    )
assert sorted(conclusion_schema_states) == (
    ["LEGACY_UNVERSIONED"] + ["LEGACY_V1"] * 7
)

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
responses = [json.loads(line) for line in completed.stdout.splitlines()]
assert len(responses) == 3
assert all("error" not in response for response in responses)
info = responses[0]["result"]["serverInfo"]
assert info["name"] == "convir-evidence-review"
assert info["version"] == "1.0.0"
assert info["sourceSha256"] == hashlib.sha256(server.read_bytes()).hexdigest()
assert info["catalogSourceSha256"] == hashlib.sha256(
    (server.parent / "convir_evidence_catalog.py").read_bytes()
).hexdigest()
assert [item["name"] for item in responses[1]["result"]["tools"]] == [
    "convir_evidence_catalog_summary", "convir_evidence_catalog_query",
]
summary = responses[2]["result"]
assert summary["isError"] is False
assert summary["structuredContent"]["catalog_sha256"] == loaded["catalog_sha256"]
assert json.loads(summary["content"][0]["text"]) == summary["structuredContent"]
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
printf 'CONVIR_EVIDENCE_REVIEW_PHASE4A_CLOUD_OK candidate=%s github_main=%s tests=%s focused_tests=%s tools=2 schema2_bindings=8 conclusion_v2=0 conclusion_legacy_v1=7 conclusion_legacy_unversioned=1 git_mutations=0 existing_experiment_runtime_state_access=0 model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$main_tip" "$test_count" "$focused_count"
