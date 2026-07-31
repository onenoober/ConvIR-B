#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'PROCESS_RELIABILITY_FIX_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-process-reliability-fix-v1
base=76f24031f34f54ef20e6ff59e65742de73af8c16
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/process-reliability-fix.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/process-reliability-fix.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'PROCESS_RELIABILITY_FIX_STAGE=checkout\n'
/usr/bin/git clone --quiet --shared --no-checkout "$seed" "$work/repo"
/usr/bin/git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(/usr/bin/git -C "$work/repo" rev-parse refs/validation/candidate)
/usr/bin/git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
/usr/bin/git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(/usr/bin/git -C "$work/repo" status --porcelain)"
/usr/bin/git -C "$work/repo" diff --check "$base" "$candidate"
/usr/bin/git -C "$work/repo" diff --quiet "$base" "$candidate" -- \
  experience_docx/experiment_logs \
  experience_docx/experiment_specs \
  experience_docx/research_programs \
  experience_docx/scientific_contracts \
  experience_docx/route_operations.json

expected=$'experience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/ROUTE_READY_FASTPATH.md\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/prepare_terminal_archive.py\nexperience_docx/tools/route_lifecycle.py\nexperience_docx/tools/route_program_api.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_prepare_terminal_archive.py\nexperience_docx/tools/tests/test_route_lifecycle.py\nexperience_docx/tools/tests/test_route_program_api.py\nexperience_docx/tools/validate_process_reliability_fix_v1_cloud.sh'
actual=$(/usr/bin/git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"

printf 'PROCESS_RELIABILITY_FIX_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tools/route_lifecycle.py" \
  "$tools/route_program_api.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_prepare_terminal_archive.py" \
  "$tests/test_route_lifecycle.py" \
  "$tests/test_route_program_api.py" \
  "$tests/test_policy_snapshot.py"

printf 'PROCESS_RELIABILITY_FIX_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
/usr/bin/git -C "$work/repo" cat-file -e "$rules_commit^{commit}"
/usr/bin/git -C "$work/repo" merge-base --is-ancestor "$rules_commit" "$candidate"
"$python" "$tools/policy_snapshot.py" --repo "$work/repo" \
  --rules-commit "$rules_commit" --check

printf 'PROCESS_RELIABILITY_FIX_STAGE=focused_regression\n'
"$python" -m unittest -v \
  test_route_program_api \
  test_route_lifecycle \
  test_convir_ops_mcp \
  test_prepare_terminal_archive \
  test_policy_snapshot

printf 'PROCESS_RELIABILITY_FIX_STAGE=full_control_plane_regression\n'
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
test "$test_count" -ge 150

printf 'PROCESS_RELIABILITY_FIX_STAGE=mcp_surface\n'
"$python" - <<'PY'
import convir_ops_mcp as ops

assert ops.SERVER_VERSION == "5.4.0"
assert ops.SCHEMA_VERSION == 4
assert set(ops.TOOLS) == {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
PY

printf 'PROCESS_RELIABILITY_FIX_CLOUD_OK candidate=%s rules_commit=%s tests=%s tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 historical_evidence_mutation=0\n' \
  "$candidate" "$rules_commit" "$test_count"
