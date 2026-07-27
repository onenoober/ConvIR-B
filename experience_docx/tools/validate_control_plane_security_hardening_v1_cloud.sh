#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-control-plane-security-hardening-v1
base=5388bbe8f075fe49d47827b32e4d78f00555c548
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-control-plane-security-hardening.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$base^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- experience_docx/experiment_logs
git -C "$work/repo" diff --check "$base" "$candidate"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/convirctl.py" \
  "$tools/experiment_spec_compiler.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_convirctl.py" \
  "$tests/test_experiment_spec_compiler.py" \
  "$tests/test_policy_snapshot.py" \
  "$tests/test_prepare_terminal_archive.py"

printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
  --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null

printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_STAGE=targeted_regression\n'
PYTHONPATH="$tools:$tests" "$python" -m unittest -v \
  test_convir_ops_mcp \
  test_convirctl \
  test_experiment_spec_compiler \
  test_policy_snapshot \
  test_prepare_terminal_archive

printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_STAGE=full_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 150

PYTHONPATH="$tools" "$python" - <<'PY'
import convir_ops_mcp as ops

assert ops.SCHEMA_VERSION == 4
assert ops.SERVER_VERSION == "5.4.0"
assert len(ops.TOOLS) == 6
assert ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS == {4, 5, 6}
PY

printf 'CONTROL_PLANE_SECURITY_HARDENING_V1_CLOUD_OK candidate=%s tests=%s tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count"
