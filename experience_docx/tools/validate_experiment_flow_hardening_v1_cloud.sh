#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'EXPERIMENT_FLOW_HARDENING_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-flow-hardening-v1
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
test -d "$runtime_root"
work=$(mktemp -d "$runtime_root/experiment-flow-hardening.XXXXXX")
case "$work" in
  /sda/home/wangyuxin/ConvIR-B/runtime/experiment-flow-hardening.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
cleanup_work() {
  case "$work" in
    /sda/home/wangyuxin/ConvIR-B/runtime/experiment-flow-hardening.*) ;;
    *) printf 'refusing cleanup outside runtime root: %s\n' "$work" >&2; return 2 ;;
  esac
  rm -rf -- "$work"
  test ! -e "$work"
}
trap cleanup_work EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" remote add github "$github"
git -C "$work/repo" fetch --quiet --no-tags github \
  "+refs/heads/$branch:refs/validation/candidate" \
  "+refs/heads/main:refs/remotes/github/main"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
printf 'EXPERIMENT_FLOW_HARDENING_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_evidence_review_mcp.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_convir_evidence_review_mcp.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_policy_snapshot.py"
bash -n "$tools/validate_experiment_flow_hardening_v1_cloud.sh"

printf 'EXPERIMENT_FLOW_HARDENING_STAGE=focused_regression\n'
PYTHONPATH="$tools:$tests" "$python" -m unittest -v \
  test_convir_evidence_review_mcp \
  test_convir_ops_mcp \
  test_policy_snapshot

printf 'EXPERIMENT_FLOW_HARDENING_STAGE=exact_candidate_completeness\n'
PYTHONPATH="$tools" "$python" - "$work/repo" "$candidate" <<'PY'
import sys

from convir_evidence_review_mcp import (
    CATALOG_SOURCE_SHA256,
    SERVER_SOURCE_SHA256,
    SERVER_VERSION,
    load_catalog_cached,
    load_legacy_registry,
    mcp_result,
    scoped_completeness_receipt,
)

repo, commit = sys.argv[1:]
registry = load_legacy_registry(repo, commit)
receipt = scoped_completeness_receipt(
    repo, commit, load_catalog_cached(repo, commit), registry,
)
assert receipt["review_completeness"] == "complete"
assert not any(receipt["unresolved_counts"].values())
assert receipt["historical_nonblocking_counts"] == {
    "registry_bound_unindexed_entries": 178,
    "registry_bound_path_only_terminal_records": 47,
    "registry_bound_ambiguous_legacy_routes": 1,
}
identity = mcp_result({"ok": True})["structuredContent"]["runtime_identity"]
assert identity["server_version"] == SERVER_VERSION == "2.2.0"
assert identity["server_source_sha256"] == SERVER_SOURCE_SHA256
assert identity["catalog_source_sha256"] == CATALOG_SOURCE_SHA256
PY

printf 'EXPERIMENT_FLOW_HARDENING_STAGE=full_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 300

printf 'EXPERIMENT_FLOW_HARDENING_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])
PY
)
"$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check

git -C "$work/repo" diff --quiet
test -z "$(git -C "$work/repo" status --porcelain)"
trap - EXIT
cleanup_work
printf 'EXPERIMENT_FLOW_HARDENING_CLOUD_OK candidate=%s tests=%s tools=6 review_tools=6 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$test_count"
