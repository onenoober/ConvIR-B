#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_EVIDENCE_REVIEW_PHASE1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-evidence-review-phase1
baseline=b096d21290dd73269c2ccba4b7389c284037f1c8
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/convir-evidence-review-phase1.XXXXXX)
case "$work" in
  /tmp/convir-evidence-review-phase1.*) ;;
  *) printf 'unexpected temporary path: %s\n' "$work" >&2; exit 2 ;;
esac
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
expected=$'experience_docx/tools/convirctl.py\nexperience_docx/tools/tests/test_convirctl.py\nexperience_docx/tools/validate_convir_evidence_review_phase1_cloud.sh'
[[ "$changed" == "$expected" ]]

tools=$work/repo/experience_docx/tools
tests=$tools/tests
"$python" -m py_compile \
  "$tools/convirctl.py" \
  "$tests/test_convirctl.py"

stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
set +e
PYTHONPATH="$tools:$tests" "$python" -m unittest discover \
  -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 120 "$stdout" >&2 || true
  tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
test_count=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ "$test_count" =~ ^[0-9]+$ ]]
test "$test_count" -ge 150

git -C "$work/repo" diff --check "$baseline" "$candidate"
git -C "$work/repo" diff --quiet
printf 'CONVIR_EVIDENCE_REVIEW_PHASE1_CLOUD_OK candidate=%s baseline=%s tests=%s model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0\n' \
  "$candidate" "$baseline" "$test_count"
