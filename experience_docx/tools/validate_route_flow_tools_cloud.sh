#!/usr/bin/env bash
set -euo pipefail

branch=main
baseline=7ae53fa6f98ffeca8c04ec8a6b21b51da5f07fd3
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/route-flow-tools.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" cat-file -e "$baseline^{commit}"
git -C "$work/repo" merge-base --is-ancestor "$baseline" "$candidate"
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
  git -C "$work/repo" diff --quiet "$baseline" "$candidate" -- "$path"
done

"$python" -m py_compile \
  "$work/repo/experience_docx/tools/validate_evidence_sync.py" \
  "$work/repo/experience_docx/tools/prepare_next_operation.py" \
  "$work/repo/experience_docx/tools/route_engineering_fixture.py" \
  "$work/repo/experience_docx/tools/build_route_asset_manifest.py" \
  "$work/repo/experience_docx/tools/tests/test_validate_evidence_sync.py" \
  "$work/repo/experience_docx/tools/tests/test_prepare_next_operation.py" \
  "$work/repo/experience_docx/tools/tests/test_route_engineering_fixture.py" \
  "$work/repo/experience_docx/tools/tests/test_build_route_asset_manifest.py"

stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
"$python" -m unittest discover \
  -s "$work/repo/experience_docx/tools/tests" -p 'test_*.py' \
  >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 120 "$stdout" >&2 || true
  tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 97

git -C "$work/repo" diff --check "$baseline" "$candidate"
git -C "$work/repo" diff --quiet
printf 'ROUTE_FLOW_TOOLS_CLOUD_OK candidate=%s baseline=%s tests=%s project_model_calls=0 synthetic_fixture_forwards=1 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$baseline" "$tests"
