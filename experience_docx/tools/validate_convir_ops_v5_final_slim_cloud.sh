#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'CONVIR_OPS_V5_FINAL_SLIM_CLOUD_FAILED line=%s command=%q rc=%s\n' "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/convir-ops-v5-final-slim-20260720
base=9b04d7dd4288f5f158c1b6d344333833fa2fce05
r16_branch=codex/haze4k-v5-r16-s3-domain-matched-action-ceiling-20260720
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
evidence_root=/sda/home/wangyuxin/ConvIR-B/runs/convir_ops_v5_final_slim_acceptance_20260720
work=$(mktemp -d /tmp/convir-ops-v5-final-slim.XXXXXX)
trap 'rm -rf -- "$work"' EXIT

printf 'CONVIR_OPS_V5_FINAL_SLIM_STAGE=checkout\n'
git clone --quiet --shared --no-checkout "$seed" "$work/repo"
git -C "$work/repo" fetch --quiet --no-tags "$github" \
  "+refs/heads/$branch:refs/validation/candidate" \
  "+refs/heads/$r16_branch:refs/validation/r16"
candidate=$(git -C "$work/repo" rev-parse refs/validation/candidate)
git -C "$work/repo" merge-base --is-ancestor "$base" "$candidate"
git -C "$work/repo" checkout --quiet --detach "$candidate"
test -z "$(git -C "$work/repo" status --porcelain)"

printf 'CONVIR_OPS_V5_FINAL_SLIM_STAGE=historical_integrity\n'
r16_card=experience_docx/experiment_cards/2026-07-20-haze4k-v5-r16-s3-domain-matched-action-ceiling.md
r16_dir=experience_docx/experiment_logs/haze4k_v5_r16_s3_domain_matched_action_ceiling_20260720
git -C "$work/repo" diff --quiet "$base" "$candidate" -- "$r16_card" "$r16_dir"
test "$(git -C "$work/repo" rev-parse "$candidate:$r16_card")" = b7c2433751255d2d08a44de4fe38d8944a82d7cd
test "$(sha256sum "$work/repo/$r16_dir/r16_s3_domain_matched_action_ceiling_closeout.json" | cut -d' ' -f1)" = a993f738d399e42bbd88101bdaeadcd4ea6f86e457be4dd5f5813157457c15d4

tools="$work/repo/experience_docx/tools"
"$python" -m py_compile \
  "$tools/convir_ops_mcp.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tools/route_lifecycle.py" \
  "$tools/route_program_api.py" \
  "$tools/route_runtime_contract.py" \
  "$tools/validate_experiment_card.py" \
  "$tools/validate_route_ready.py" \
  "$tools/tests/test_convir_ops_v5_final_slim.py"

printf 'CONVIR_OPS_V5_FINAL_SLIM_STAGE=full_regression\n'
stdout="$work/unittest.stdout"
stderr="$work/unittest.stderr"
set +e
PYTHONPATH="$tools" "$python" -m unittest discover \
  -s "$tools/tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  tail -n 200 "$stdout" >&2 || true
  tail -n 200 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]
test "$tests" -ge 120

printf 'CONVIR_OPS_V5_FINAL_SLIM_STAGE=historical_archive_audit\n'
PYTHONPATH="$tools" "$python" "$tools/prepare_terminal_archive.py" \
  --source-repo "$work/repo" --source-ref "$candidate" \
  --route-id haze4k_v5_r16_s3_domain_matched_action_ceiling_20260720 \
  --closeout "$r16_dir/r16_s3_domain_matched_action_ceiling_closeout.json" \
  --contract "$r16_card" \
  --conclusion "$r16_dir/r16_s3_scientific_conclusion.json" \
  --receipt b09e5bbe031b7581bc1c0955dab055eea93e9419a3b7472654e853b7f30c3e2a \
  --audit-only --existing-archive --local-evidence-only \
  --report "$work/r16-audit.json" >/dev/null

printf 'CONVIR_OPS_V5_FINAL_SLIM_STAGE=surface_and_cost\n'
PYTHONPATH="$tools" "$python" - "$work/acceptance.json" "$tests" "$candidate" <<'PY'
import json, sys
import convir_ops_mcp as ops

output, tests, candidate = sys.argv[1:]
assert ops.SERVER_VERSION == "5.0.0"
assert ops.SCHEMA_VERSION == 4
assert len(ops.TOOLS) == 6
assert set(ops.TOOLS) == {
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
payload = {
    "operation_state": "READY", "ok": True,
    "changed_paths": [f"path-{index}" for index in range(1000)],
}
result = ops.text_result(json.dumps(payload), structured=payload)
summary_bytes = len(result["content"][0]["text"].encode())
assert summary_bytes <= 512
assert len(result["structuredContent"]["changed_paths"]) == 1000
json.dump({
    "schema_version": 1,
    "status": "COMPLETED_GATE_PASS",
    "decision": "CONVIR_OPS_V5_FINAL_SLIM_ADOPTION",
    "candidate_commit": candidate,
    "tests_passed": int(tests),
    "mcp_protocol_schema": ops.SCHEMA_VERSION,
    "manifest_schemas": sorted(ops.SUPPORTED_MANIFEST_SCHEMA_VERSIONS),
    "tool_count": len(ops.TOOLS),
    "default_summary_bytes": summary_bytes,
    "historical_r16_audit": "PASS",
    "old_evidence_modified": False,
    "model_calls": 0,
    "gpu_access": 0,
    "dataset_access": 0,
    "checkpoint_access": 0,
    "confirmation_access": 0,
    "canary_access": 0,
    "locked_test_access": 0,
}, open(output, "w"), indent=2, sort_keys=True)
PY

mkdir -p "$evidence_root"
cp "$work/acceptance.json" "$evidence_root/acceptance-${candidate}.json"
cp "$work/r16-audit.json" "$evidence_root/r16-compatibility-${candidate}.json"
printf 'CONVIR_OPS_V5_FINAL_SLIM_CLOUD_OK candidate=%s tests=%s tools=6 summary_max=512 model_calls=0 gpu_access=0 protected_data_access=0\n' \
  "$candidate" "$tests"
