#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-assistant-slim-v1
base=39205b2f6a7e7dd79ef1a3a0d9c1c491bfba41f5
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/experiment-assistant-contract-v1.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/experiment-assistant-contract-v1.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_STAGE=checkout\n'
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
  experience_docx/engineering_failures \
  experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl \
  experience_docx/route_operations.json

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/experiment_assistant_contract.py" \
  "$tests/test_experiment_assistant_contract.py"

printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_STAGE=tests\n'
output=$work/unittest.txt
"$python" -m unittest -v test_experiment_assistant_contract >"$output" 2>&1
/bin/cat "$output"
test_count=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$output" | /usr/bin/tail -n 1)
test "$test_count" -ge 12

printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_STAGE=surface\n'
"$python" - <<'PY'
import experiment_assistant_contract as assistant

assert assistant.MAX_AUTOMATIC_REPAIRS == 2
assert assistant.PUBLIC_TOOL_NAMES == (
    "experiment_start", "experiment_status", "experiment_cancel",
    "experiment_repair", "experiment_get", "experiment_search",
)
serialized = repr(assistant.PUBLIC_TOOL_SCHEMAS)
for forbidden in (
    "plan_token", "receipt", "catalog_sha256", "inventory_sha256",
    "snapshot_commit", "route_branch_commit", "schema_version",
):
    assert forbidden not in serialized, forbidden
print("EXPERIMENT_ASSISTANT_CONTRACT_V1_SURFACE_OK")
PY

printf 'EXPERIMENT_ASSISTANT_CONTRACT_V1_CLOUD_OK candidate=%s tests=%s gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$test_count"
