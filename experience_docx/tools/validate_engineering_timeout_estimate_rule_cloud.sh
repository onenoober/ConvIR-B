#!/usr/bin/env bash
set -euo pipefail

branch=codex/engineering-timeout-estimate-rule
base=0a7a1762065cd7418e1966a58a2e06243d44d440
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(mktemp -d /tmp/engineering-timeout-estimate-rule.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"
git -C "$work/repo" diff --check "$base" "$candidate"
git -C "$work/repo" diff --quiet "$base" "$candidate" -- experience_docx/experiment_logs

changed=$(git -C "$work/repo" diff --name-only "$base" "$candidate")
rules_only=$'experience_docx/RULE_COMPATIBILITY.json\nexperience_docx/tools/route_runtime_contract.py\nexperience_docx/tools/tests/test_route_runtime_contract.py\nexperience_docx/tools/validate_engineering_timeout_estimate_rule_cloud.sh'
with_snapshot=$'experience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/RULE_COMPATIBILITY.json\nexperience_docx/tools/route_runtime_contract.py\nexperience_docx/tools/tests/test_route_runtime_contract.py\nexperience_docx/tools/validate_engineering_timeout_estimate_rule_cloud.sh'
if [[ "$changed" != "$rules_only" && "$changed" != "$with_snapshot" ]]; then
  printf 'unexpected changed paths:\n%s\n' "$changed" >&2
  exit 1
fi

tools=$work/repo/experience_docx/tools
tests=$tools/tests
"$python" -m py_compile \
  "$tools/route_runtime_contract.py" \
  "$tests/test_route_runtime_contract.py"
"$python" -m json.tool \
  "$work/repo/experience_docx/RULE_COMPATIBILITY.json" >/dev/null
PYTHONPATH="$tools:$tests" "$python" -m unittest test_route_runtime_contract

PYTHONPATH="$tools" "$python" - \
  "$work/repo/experience_docx/RULE_COMPATIBILITY.json" <<'PY'
import json
import sys

import convir_ops_mcp

with open(sys.argv[1], encoding="utf-8") as handle:
    compatibility = json.load(handle)
assert compatibility["compatibility_id"] == "science-fastpath-contract-v4"
assert "c35c636a5616df27e4a53764aeada9094478cb11" in \
    compatibility["compatible_prior_rules_commits"]
assert convir_ops_mcp.SERVER_VERSION == "5.9.0"
PY

if [[ "$changed" == "$with_snapshot" ]]; then
  rules_commit=$("$python" - \
    "$work/repo/experience_docx/AI_POLICY_SNAPSHOT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["rules_commit"])
PY
  )
  test "$rules_commit" = "$(git -C "$work/repo" rev-parse "$candidate^")"
  PYTHONPATH="$tools" "$python" "$tools/policy_snapshot.py" \
    --repo "$work/repo" --rules-commit "$rules_commit" --check >/dev/null
fi

printf 'ENGINEERING_TIMEOUT_ESTIMATE_RULE_CLOUD_OK candidate=%s snapshot=%s model_calls=0 gpu_access=0 dataset_access=0 protected_data_access=0 experiment_launches=0\n' \
  "$candidate" "$([[ "$changed" == "$with_snapshot" ]] && printf checked || printf pending)"
