#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'ROUTE_AUTHORING_FASTPATH_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=main
baseline=39eded234afb890946c189599009ffe1b81b90c9
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/route-authoring-fastpath.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$baseline^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
main=$baseline
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

unchanged=(
  experience_docx/tools/convir_ops_mcp.py
  experience_docx/tools/run_route_operation.sh
  experience_docx/tools/route_lifecycle.py
  experience_docx/tools/route_program_api.py
  experience_docx/tools/route_runtime_contract.py
  experience_docx/tools/run_telemetry.py
)
for path in "${unchanged[@]}"; do
  git -C "$work/repo" diff --quiet "$main" "$candidate" -- "$path"
done

"$python" -m py_compile \
  "$work/repo/experience_docx/tools/validate_experiment_card.py" \
  "$work/repo/experience_docx/tools/validate_route_ready.py" \
  "$work/repo/experience_docx/tools/tests/test_validate_experiment_card.py" \
  "$work/repo/experience_docx/tools/tests/test_validate_route_ready.py"

stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
"$python" -m unittest discover \
  -s "$work/repo/experience_docx/tools/tests" -p 'test_*.py' \
  >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 100 "$stdout" >&2 || true
  tail -n 100 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]

git -C "$work/repo" diff --check "$main" "$candidate"
git -C "$work/repo" diff --quiet
printf 'ROUTE_AUTHORING_FASTPATH_CLOUD_OK candidate=%s main=%s tests=%s model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$main" "$tests"
