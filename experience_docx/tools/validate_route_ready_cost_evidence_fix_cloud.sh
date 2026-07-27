#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'ROUTE_READY_COST_EVIDENCE_FIX_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-route-ready-cost-evidence-schema-fix-v1
baseline=38f7aebf63d6b0611c31115c27c89c7b01c31008
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/route-ready-cost-evidence-fix.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$baseline^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

changed=$(git -C "$work/repo" diff --name-only "$baseline" "$candidate")
expected=$'experience_docx/tools/tests/test_validate_route_ready.py\nexperience_docx/tools/validate_route_ready.py\nexperience_docx/tools/validate_route_ready_cost_evidence_fix_cloud.sh'
[[ "$changed" == "$expected" ]]

"$python" -m py_compile \
  "$work/repo/experience_docx/tools/validate_route_ready.py" \
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

git -C "$work/repo" diff --check "$baseline" "$candidate"
git -C "$work/repo" diff --quiet
printf 'ROUTE_READY_COST_EVIDENCE_FIX_CLOUD_OK candidate=%s baseline=%s tests=%s model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$baseline" "$tests"
