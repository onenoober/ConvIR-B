#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'REVIEW_FACTS_RECOVERY_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/daytime-dehazing-spatially-adaptive-restoration-v1
base=86653086060c4600c5f3b84f6937c05c2576f737
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/review-facts-recovery.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/review-facts-recovery.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'REVIEW_FACTS_RECOVERY_STAGE=checkout\n'
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

expected=$'experience_docx/AI_POLICY_SNAPSHOT.json\nexperience_docx/CONVIR_EVIDENCE_REVIEW.md\nexperience_docx/SCIENCE_FASTPATH.md\nexperience_docx/tools/convir_evidence_catalog.py\nexperience_docx/tools/convir_evidence_review_mcp.py\nexperience_docx/tools/prepare_terminal_archive.py\nexperience_docx/tools/tests/test_convir_evidence_catalog.py\nexperience_docx/tools/tests/test_convir_evidence_review_mcp.py\nexperience_docx/tools/tests/test_prepare_terminal_archive.py\nexperience_docx/tools/validate_review_facts_recovery_v1_cloud.sh'
actual=$(/usr/bin/git -C "$work/repo" diff --name-only "$base" "$candidate")
test "$actual" = "$expected"

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"

printf 'REVIEW_FACTS_RECOVERY_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/convir_evidence_catalog.py" \
  "$tools/convir_evidence_review_mcp.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tools/policy_snapshot.py" \
  "$tests/test_convir_evidence_catalog.py" \
  "$tests/test_convir_evidence_review_mcp.py" \
  "$tests/test_prepare_terminal_archive.py" \
  "$tests/test_policy_snapshot.py"

printf 'REVIEW_FACTS_RECOVERY_STAGE=policy_snapshot\n'
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

printf 'REVIEW_FACTS_RECOVERY_STAGE=focused_regression\n'
"$python" -m unittest -v \
  test_prepare_terminal_archive \
  test_convir_evidence_catalog \
  test_convir_evidence_review_mcp \
  test_policy_snapshot

printf 'REVIEW_FACTS_RECOVERY_STAGE=full_control_plane_regression\n'
stdout=$work/unittest.stdout
stderr=$work/unittest.stderr
trap - ERR
set +e
"$python" -m unittest discover -s "$tests" -p 'test_*.py' >"$stdout" 2>"$stderr"
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
test "$test_count" -ge 150

printf 'REVIEW_FACTS_RECOVERY_STAGE=fresh_mcp_surface\n'
"$python" - "$tools/convir_evidence_review_mcp.py" <<'PY'
import json
import subprocess
import sys

server = sys.argv[1]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "review-facts-recovery", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]
payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
completed = subprocess.run(
    [sys.executable, server], input=payload, text=True, capture_output=True,
    timeout=30, check=True,
)
assert completed.stderr.strip() == "", completed.stderr
responses = {item["id"]: item for item in map(json.loads, completed.stdout.splitlines())}
expected_tools = {
    "convir_evidence_catalog_summary",
    "convir_evidence_completeness_receipt",
    "convir_evidence_catalog_query",
    "convir_evidence_bundle",
    "convir_evidence_cloud_inventory_summary",
    "convir_evidence_cloud_inventory_query",
    "convir_evidence_cloud_text_read",
}
assert set(responses) == {1, 2}, responses
assert responses[1]["result"]["serverInfo"]["version"] == "1.5.0"
assert {item["name"] for item in responses[2]["result"]["tools"]} == expected_tools
PY

printf 'REVIEW_FACTS_RECOVERY_CLOUD_OK candidate=%s rules_commit=%s tests=%s tools=7 gpu_access=0 dataset_access=0 protected_data_access=0 historical_evidence_mutation=0\n' \
  "$candidate" "$rules_commit" "$test_count"
