#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/finalization-unshallow-history-fix
base=f9a1e6aa5d43b0f1440386a90c4434889fff190c
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/finalization-unshallow-history-fix.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/finalization-unshallow-history-fix.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_STAGE=checkout\n'
/usr/bin/git clone --quiet --shared --no-checkout "$seed" "$work/repo"
/usr/bin/git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(/usr/bin/git -C "$work/repo" rev-parse refs/validation/candidate)
/usr/bin/git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
/usr/bin/git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(/usr/bin/git -C "$work/repo" status --porcelain)"
/usr/bin/git -C "$work/repo" diff --check "$base" "$candidate"

expected=$'experience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/validate_finalization_unshallow_history_fix_cloud.sh'
actual=$(/usr/bin/git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tests/test_convir_ops_mcp.py"

printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_STAGE=focused_regression\n'
"$python" -m unittest -v test_convir_ops_mcp

printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_STAGE=full_control_plane_regression\n'
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

printf 'FINALIZATION_UNSHALLOW_HISTORY_FIX_CLOUD_OK candidate=%s tests=%s gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launch=0\n' \
  "$candidate" "$test_count"
