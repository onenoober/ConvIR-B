#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/science-loop-schema3-hardening-v1
base=189db2e71fe655e956dc3dfdf61df19a50835be3
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/science-loop-schema3-hardening-v1.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/science-loop-schema3-hardening-v1.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=checkout\n'
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

expected=$'AGENTS.md\nexperience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/CONVIR_OPS_MCP.md\nexperience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md\nexperience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md\nexperience_docx/ROUTE_READY_FASTPATH.md\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_evidence_review_mcp.py\nexperience_docx/tools/convir_ops_mcp.py\nexperience_docx/tools/experiment_spec_compiler.py\nexperience_docx/tools/policy_snapshot.py\nexperience_docx/tools/scientific_contract.py\nexperience_docx/tools/tests/test_convir_evidence_review_mcp.py\nexperience_docx/tools/tests/test_convir_ops_mcp.py\nexperience_docx/tools/tests/test_policy_snapshot.py\nexperience_docx/tools/tests/test_scientific_contract.py\nexperience_docx/tools/tests/test_validate_route_ready.py\nexperience_docx/tools/validate_route_ready.py\nexperience_docx/tools/validate_science_loop_schema3_hardening_v1_cloud.sh'
actual=$(/usr/bin/git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/scientific_contract.py" \
  "$tools/experiment_spec_compiler.py" \
  "$tools/convir_ops_mcp.py" \
  "$tools/convir_evidence_review_mcp.py" \
  "$tools/validate_route_ready.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_scientific_contract.py" \
  "$tests/test_experiment_spec_compiler.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_convir_evidence_review_mcp.py" \
  "$tests/test_validate_route_ready.py" \
  "$tests/test_policy_snapshot.py"

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=policy_snapshot\n'
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

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=focused_regression\n'
"$python" -m unittest -v \
  test_scientific_contract \
  test_experiment_spec_compiler \
  test_convir_evidence_catalog \
  test_convir_evidence_review_mcp \
  test_convir_ops_mcp \
  test_route_lifecycle \
  test_validate_route_ready \
  test_policy_snapshot

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=full_control_plane_regression\n'
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
test "$test_count" -ge 380

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_STAGE=machine_contract\n'
"$python" - "$work/repo" <<'PY'
import json
import sys
from pathlib import Path

import convir_evidence_review_mcp as review
import convir_ops_mcp as ops
import validate_route_ready as ready

repo = Path(sys.argv[1])
assert ops.SERVER_VERSION == "5.9.0"
assert ops.SCHEMA_VERSION == 4
assert len(ops.TOOLS) == 6
assert review.SERVER_VERSION == "2.2.0"
assert len(review.TOOLS) == 6
assert "catalog_sha256" in review.TOOLS[
    "convir_evidence_catalog_query"
]["inputSchema"]["properties"]
ready.require_current_runnable_schema(6, {"schema_version": 3})
try:
    ready.require_current_runnable_schema(5)
except ready.ReadyError:
    pass
else:
    raise AssertionError("historical manifest unexpectedly remained runnable")
with (repo / "experience_docx/AI_POLICY_SNAPSHOT.json").open(encoding="utf-8") as handle:
    snapshot = json.load(handle)
assert "experience_docx/EXPERIMENT_GOVERNANCE_PROTOCOL.md" in snapshot[
    "change_routes"
]["read_only_audit"]["read_full"]
with (repo / "experience_docx/RULE_COMPATIBILITY.json").open(encoding="utf-8") as handle:
    compatibility = json.load(handle)
assert compatibility == {
    "schema_version": 1,
    "compatibility_id": "science-fastpath-contract-v4",
    "compatible_prior_rules_commits": [],
}
route_ready = (repo / "experience_docx/ROUTE_READY_FASTPATH.md").read_text(
    encoding="utf-8"
)
model_ops = (repo / "experience_docx/MODEL_RUN_OPERATIONS_PROTOCOL.md").read_text(
    encoding="utf-8"
)
assert "one schema-v2 experiment spec" not in route_ready
assert "New scientific schema-2 entrypoints" not in model_ops
PY

printf 'SCIENCE_LOOP_SCHEMA3_HARDENING_V1_CLOUD_OK candidate=%s rules_commit=%s tests=%s ops_version=5.9.0 review_version=2.2.0 ops_tools=6 review_tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launch=0 historical_evidence_mutation=0\n' \
  "$candidate" "$rules_commit" "$test_count"
