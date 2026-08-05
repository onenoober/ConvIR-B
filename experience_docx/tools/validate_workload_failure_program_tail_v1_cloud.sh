#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/workload-failure-program-tail-v1
base=1c5f611f65e21f1f737241209aeb9de72f26e8a4
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/workload-failure-program-tail-v1.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/workload-failure-program-tail-v1.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_STAGE=checkout\n'
/usr/bin/git clone --quiet --shared --no-checkout "$seed" "$work/repo"
/usr/bin/git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(/usr/bin/git -C "$work/repo" rev-parse refs/validation/candidate)
/usr/bin/git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
/usr/bin/git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(/usr/bin/git -C "$work/repo" status --porcelain)"
/usr/bin/git -C "$work/repo" diff --check "$base" "$candidate"

expected=$'experience_docx/tools/route_lifecycle.py\nexperience_docx/tools/tests/test_route_lifecycle.py\nexperience_docx/tools/validate_workload_failure_program_tail_v1_cloud.sh'
actual=$(/usr/bin/git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/route_lifecycle.py" \
  "$tests/test_route_lifecycle.py"

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_STAGE=policy_snapshot\n'
rules_commit=$("$python" - "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
)
/usr/bin/git -C "$work/repo" cat-file -e "$rules_commit^{commit}"
"$python" "$tools/policy_snapshot.py" --repo "$work/repo" \
  --rules-commit "$rules_commit" --check

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_STAGE=focused_regression\n'
"$python" -m unittest -v test_route_lifecycle

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_STAGE=full_control_plane_regression\n'
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

printf 'WORKLOAD_FAILURE_PROGRAM_TAIL_V1_CLOUD_OK candidate=%s rules_commit=%s tests=%s gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launch=0\n' \
  "$candidate" "$rules_commit" "$test_count"
