#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'WORKFLOW_COORDINATION_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-workflow-coordination-v1
base=b4f8ab82726526fdddf9973ecec641cf3b67c0d2
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/workflow-coordination-v1.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

tools="$work/repo/experience_docx/tools"
"$python" -m py_compile \
  "$tools/experiment_spec_compiler.py" \
  "$tools/validate_route_ready.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/tests/test_experiment_spec_compiler.py" \
  "$tools/tests/test_validate_route_ready.py" \
  "$tools/tests/test_convir_ops_v5_final_slim.py"

stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
PYTHONPATH="$tools" "$python" -m unittest discover \
  -s "$tools/tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 300

policy_rules_commit=$("$python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])' \
  "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json")
[[ $policy_rules_commit =~ ^[0-9a-f]{40}$ ]]
"$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$policy_rules_commit" --check

PYTHONPATH="$tools" "$python" - <<'PY'
import convir_ops_mcp as ops

assert ops.SERVER_VERSION == "5.5.0"
assert ops.SCHEMA_VERSION == 4
assert len(ops.TOOLS) == 6
assert set(ops.TOOLS) == {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
receipt = ops.build_snapshot_phase_receipt({
    "branch": "codex/fixture", "head": "a" * 40,
    "github_main_remote": "b" * 40,
    "github_main_ref_fresh": True, "worktree_clean": True,
    "authoritative_snapshot": {
        "status": "NO_TERMINAL_RECORD", "route_id": "fixture",
    },
})
assert receipt["scientific_authorization"] == "NOT_DERIVED"
assert receipt["allowed_next_action"] == "resolve_route_identity_or_snapshot"
assert len(receipt["receipt_sha256"]) == 64
PY

git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet
printf 'WORKFLOW_COORDINATION_V1_CLOUD_OK candidate=%s tests=%s tools=6 model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
