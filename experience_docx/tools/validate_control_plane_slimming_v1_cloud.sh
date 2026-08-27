#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONTROL_PLANE_SLIMMING_V1_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-control-plane-slimming-v1
base=951d05cc9f8d202ad4f6571e198610a15576e5c8
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/convir-control-plane-slimming-v1.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/convir-control-plane-slimming-v1.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'CONTROL_PLANE_SLIMMING_V1_STAGE=checkout\n'
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
  experience_docx/experiment_specs \
  experience_docx/research_programs \
  experience_docx/scientific_contracts \
  experience_docx/route_operations.json

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'CONTROL_PLANE_SLIMMING_V1_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/experiment_spec_compiler.py" \
  "$tools/scientific_contract.py" \
  "$tools/validate_route_ready.py" \
  "$tests/test_convir_ops_mcp.py" \
  "$tests/test_convir_ops_v5_final_slim.py" \
  "$tests/test_experiment_spec_compiler.py" \
  "$tests/test_scientific_contract.py" \
  "$tests/test_validate_route_ready.py"

printf 'CONTROL_PLANE_SLIMMING_V1_STAGE=policy_snapshot\n'
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
  --rules-commit "$rules_commit" --check >/dev/null

printf 'CONTROL_PLANE_SLIMMING_V1_STAGE=full_regression\n'
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
test "$test_count" -ge 400

printf 'CONTROL_PLANE_SLIMMING_V1_STAGE=machine_contract\n'
"$python" - "$work/repo" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

import convir_ops_mcp as ops
import experiment_spec_compiler as compiler
import validate_route_ready as ready
from test_experiment_spec_compiler import sources_v3

repo = Path(sys.argv[1])
assert ops.SCHEMA_VERSION == 4
assert len(ops.TOOLS) == 6
assert ops.SERVER_SOURCE_SHA256 == hashlib.sha256(
    (repo / "experience_docx/tools/convir_ops_mcp.py").read_bytes()
).hexdigest()

compatibility = json.loads(
    (repo / "experience_docx/RULE_COMPATIBILITY.json").read_text(encoding="utf-8")
)
assert compatibility == {
    "schema_version": 2,
    "compatibility_id": "science-fastpath-contract-v5",
    "compatible_prior_ids": ["science-fastpath-contract-v4"],
}

program, full = sources_v3()
slim = json.loads(json.dumps(full))
scientific = slim["operations"]["ACCEPT"]["scientific_contract"]
scientific.pop("research_update_binding")
scientific["decision_table"] = {
    "terminal_actions": scientific["decision_table"]["terminal_actions"],
    "policy": "typed_gate_precedence_v1",
}
slim.pop("rules_commit")
slim.pop("first_operation")
full_bytes = len(compiler.json_bytes(full))
slim_bytes = len(compiler.json_bytes(slim))
# Runtime, asset, capability and precision contracts remain invariant in this total.
minimum_source_reduction_pct = 30
assert slim_bytes * 100 <= full_bytes * (100 - minimum_source_reduction_pct), (
    full_bytes, slim_bytes,
)
bundle = compiler.compile_bundle(
    spec_relpath="experience_docx/experiment_specs/final_slim.json",
    spec_raw=compiler.json_bytes(slim),
    program_raw=compiler.json_bytes(program),
    evidence_exists=lambda _: True,
    authoritative_snapshot_commit="a" * 40,
    read_authoritative_file=lambda _: b"",
)
scientific_output = json.loads(
    bundle["experience_docx/scientific_contracts/final_slim__ACCEPT.json"]
)
assert scientific_output["schema_version"] == 2
assert scientific_output["decision_table"] == {
    "terminal_actions": scientific["decision_table"]["terminal_actions"],
    "policy": "typed_gate_precedence_v1",
}
contract = {"schema_version": 2}
assert not ready.requires_completed_unit_ledger(
    contract, {"resume_policy": "none", "total_units": 10},
)
assert ready.requires_completed_unit_ledger(
    contract, {"resume_policy": "complete_units", "total_units": 10},
)

governance_paths = [
    "AGENTS.md",
    "experience_docx/CONVIR_OPS_MCP.md",
    "experience_docx/CONVIR_EVIDENCE_REVIEW.md",
]
pin = re.compile(r"convir-ops.{0,40}(?:version|v)\s*`?\d+\.\d+\.\d+", re.I)
for relpath in governance_paths:
    assert not pin.search((repo / relpath).read_text(encoding="utf-8")), relpath

print(json.dumps({
    "protocol_schema": ops.SCHEMA_VERSION,
    "tool_count": len(ops.TOOLS),
    "full_source_bytes": full_bytes,
    "slim_source_bytes": slim_bytes,
    "minimum_source_reduction_pct": minimum_source_reduction_pct,
    "source_reduction_pct": round((1 - slim_bytes / full_bytes) * 100, 1),
    "scientific_output_schema": scientific_output["schema_version"],
    "compatibility_schema": compatibility["schema_version"],
    "semver_governance_gate": False,
    "project_completeness_route_prerequisite": False,
    "gpu_access": 0,
    "dataset_access": 0,
    "protected_data_access": 0,
    "experiment_launches": 0,
}, sort_keys=True))
PY

printf 'CONTROL_PLANE_SLIMMING_V1_CLOUD_OK candidate=%s rules_commit=%s tests=%s gpu_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$rules_commit" "$test_count"
