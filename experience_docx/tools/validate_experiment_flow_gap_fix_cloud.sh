#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'EXPERIMENT_FLOW_GAP_FIX_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-flow-hardening-v1
base=aeb1945b9dd439e190532711057cf127f819cadb
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/experiment-flow-gap-fix.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/experiment-flow-gap-fix.*) /bin/rm -rf -- "$work" ;;
    *) printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2 ;;
  esac
}
trap cleanup EXIT

printf 'EXPERIMENT_FLOW_GAP_FIX_STAGE=checkout\n'
/usr/bin/git clone --quiet --shared --no-checkout "$seed" "$work/repo"
/usr/bin/git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(/usr/bin/git -C "$work/repo" rev-parse refs/validation/candidate)
/usr/bin/git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
/usr/bin/git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(/usr/bin/git -C "$work/repo" status --porcelain)"
/usr/bin/git -C "$work/repo" diff --check "$base" "$candidate"
/usr/bin/git -C "$work/repo" diff --quiet "$base" "$candidate" -- \
  experience_docx/tools/convir_ops_mcp.py \
  experience_docx/tools/convir_evidence_review_mcp.py

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'EXPERIMENT_FLOW_GAP_FIX_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/experiment_spec_compiler.py" \
  "$tools/research_program_contract.py" \
  "$tools/route_program_api.py" \
  "$tools/validate_evidence_sync.py" \
  "$tools/validate_route_ready.py" \
  "$tests/test_research_program_contract.py" \
  "$tests/test_route_program_api.py" \
  "$tests/test_validate_evidence_sync.py" \
  "$tests/test_validate_route_ready.py"

printf 'EXPERIMENT_FLOW_GAP_FIX_STAGE=policy_snapshot\n'
snapshot=$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json
snapshot_commit=$("$python" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["rules_commit"])' \
  "$snapshot")
set +e
"$python" "$tools/policy_snapshot.py" --repo "$work/repo" \
  --rules-commit "$snapshot_commit" --check >/dev/null 2>&1
snapshot_rc=$?
set -e
if [[ $snapshot_rc -ne 0 ]]; then
  "$python" "$tools/policy_snapshot.py" --repo "$work/repo" \
    --rules-commit "$candidate" --write >/dev/null
  "$python" "$tools/policy_snapshot.py" --repo "$work/repo" \
    --rules-commit "$candidate" --check >/dev/null
  echo AI_POLICY_SNAPSHOT_JSON_BEGIN
  /bin/cat "$snapshot"
  echo AI_POLICY_SNAPSHOT_JSON_END
else
  printf 'AI_POLICY_SNAPSHOT_CURRENT rules_commit=%s\n' "$snapshot_commit"
fi

printf 'EXPERIMENT_FLOW_GAP_FIX_STAGE=focused_regression\n'
"$python" -m unittest -v \
  test_research_program_contract \
  test_route_program_api \
  test_validate_evidence_sync \
  test_validate_route_ready

printf 'EXPERIMENT_FLOW_GAP_FIX_STAGE=full_control_plane_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
"$python" -m unittest discover -s "$tests" -p 'test_*.py' \
  >"$stdout" 2>"$stderr"
rc=$?
set -e
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
if [[ $rc -ne 0 ]]; then
  /usr/bin/tail -n 200 "$stdout" >&2 || true
  /usr/bin/tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | /usr/bin/tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 300

printf 'EXPERIMENT_FLOW_GAP_FIX_CLOUD_OK candidate=%s tests=%s gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launch=0\n' \
  "$candidate" "$test_count"
