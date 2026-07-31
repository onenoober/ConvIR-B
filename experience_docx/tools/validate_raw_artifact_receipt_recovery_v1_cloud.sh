#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/raw-artifact-receipt-recovery-v1
base=75417985cb16108b20166bcd1f9948be0d755eb9
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/raw-artifact-receipt-recovery.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/raw-artifact-receipt-recovery.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=checkout\n'
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

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/route_lifecycle.py" \
  "$tools/prepare_terminal_archive.py" \
  "$tests/test_route_lifecycle.py" \
  "$tests/test_prepare_terminal_archive.py"

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=focused_regression\n'
"$python" -m unittest -v test_route_lifecycle test_prepare_terminal_archive

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=cloud_manifest_recovery\n'
"$python" - "$work" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import prepare_terminal_archive as archive

root = Path(sys.argv[1]) / "manifest-fixture"
evidence = root / "evidence"
evidence.mkdir(parents=True)
run_root = root / "runs"
output = run_root / "a1-r1"
control = output / "control"
control.mkdir(parents=True)
closeout_name = "a1_closeout.json"
receipt_name = "a1_raw_artifact_receipt.json"
closeout_raw = b'{"terminal":"fixed"}\n'
(evidence / closeout_name).write_bytes(closeout_raw)
rows = [
    {
        "schema_version": 2,
        "relative_path": "contract/check.json",
        "artifact_class": "contract_output",
        "bytes": 2,
        "sha256": "1" * 64,
    },
    {
        "schema_version": 2,
        "relative_path": "workload/summary.json",
        "artifact_class": "workload_output",
        "bytes": 3,
        "sha256": "2" * 64,
    },
    {
        "schema_version": 2,
        "relative_path": "workload/units/unit-1.json",
        "artifact_class": "workload/units_output",
        "bytes": 4,
        "sha256": "3" * 64,
    },
]
manifest_raw = b"".join(
    (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
    for item in rows
)
manifest = control / "raw_artifact_manifest.jsonl"
manifest.write_bytes(manifest_raw)
receipt = {
    "schema_version": 2,
    "route_id": "route", "operation_id": "A1", "run_id": "a1-r1",
    "route_commit": "a" * 40,
    "manifest_relative_path": archive.RAW_ARTIFACT_MANIFEST_RELPATH,
    "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    "entry_count": 3,
    "total_bytes": 9,
    "category_counts": {"contract_output": 1, "workload_output": 1},
    "scope_roots": archive.RAW_ARTIFACT_SCOPE_ROOTS,
    "excluded_paths": archive.RAW_ARTIFACT_EXCLUDED_PATHS,
}
receipt_raw = json.dumps(receipt, sort_keys=True).encode()
(evidence / receipt_name).write_bytes(receipt_raw)
context = {
    "evidence_dir": str(evidence),
    "run_root": str(run_root),
    "output_path": str(output),
    "output_id": "a1-r1",
    "validated_closeout_filename": closeout_name,
    "validated_closeout_sha256": hashlib.sha256(closeout_raw).hexdigest(),
}
body = archive.raw_artifact_manifest_recovery_body(
    context, receipt_name, hashlib.sha256(receipt_raw).hexdigest(), receipt,
)
completed = subprocess.run(
    ["/bin/bash"], input=body, text=True, capture_output=True,
    timeout=30, check=True,
)
summary = archive.parse_raw_artifact_manifest_summary(completed.stdout)
assert summary["recovered_category_counts"] == {
    "contract_output": 1, "workload_output": 2,
}
assert summary["misclassified_nested_entry_count"] == 1
manifest.write_bytes(manifest_raw + b"{}\n")
tampered = subprocess.run(
    ["/bin/bash"], input=body, text=True, capture_output=True,
    timeout=30, check=False,
)
assert tampered.returncode != 0
print("RAW_ARTIFACT_RECEIPT_RECOVERY_MANIFEST_FIXTURE_OK")
PY

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=full_control_plane_regression\n'
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

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_STAGE=fresh_mcp_surface\n'
"$python" - "$tools/convir_ops_mcp.py" <<'PY'
import json
import subprocess
import sys

server = sys.argv[1]
requests = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "raw-artifact-recovery", "version": "1.0"},
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
    "convir_route_plan", "convir_route_start", "convir_route_finish",
    "convir_evidence_list", "convir_evidence_fetch", "convir_git_status",
}
assert set(responses) == {1, 2}, responses
assert responses[1]["result"]["serverInfo"]["version"] == "5.4.0"
assert {item["name"] for item in responses[2]["result"]["tools"]} == expected_tools
PY

printf 'RAW_ARTIFACT_RECEIPT_RECOVERY_CLOUD_OK candidate=%s tests=%s tools=6 gpu_access=0 dataset_access=0 protected_data_access=0 historical_evidence_mutation=0\n' \
  "$candidate" "$test_count"
