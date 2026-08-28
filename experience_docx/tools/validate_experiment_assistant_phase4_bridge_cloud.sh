#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  if [[ -n ${output:-} && -f $output ]]; then
    /bin/cat "$output" >&2
  fi
  printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-assistant-slim-v1
base=533dced04e129e2a3e2f30fc61b42f5f21bc67a4
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/experiment-assistant-phase4-bridge.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/experiment-assistant-phase4-bridge.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_STAGE=checkout\n'
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
  experience_docx/experiment_records \
  experience_docx/EXPERIMENT_RECORD_INDEX.jsonl \
  experience_docx/EXPERIMENT_TERMINAL_INDEX.jsonl \
  experience_docx/route_operations.json \
  experience_docx/AI_POLICY_SNAPSHOT.json \
  AGENTS.md

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/experiment_assistant_contract.py" \
  "$tools/experiment_assistant_datasets.py" \
  "$tools/experiment_assistant_archive.py" \
  "$tools/experiment_assistant_snapshot.py" \
  "$tools/experiment_assistant_runner.py" \
  "$tools/experiment_assistant_transport.py" \
  "$tools/convir_experiment_assistant_mcp.py" \
  "$tests/test_experiment_assistant_contract.py" \
  "$tests/test_experiment_assistant_datasets.py" \
  "$tests/test_experiment_assistant_snapshot.py" \
  "$tests/test_experiment_assistant_runner.py"

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_STAGE=tests\n'
output=$work/unittest.txt
"$python" -m unittest -v \
  test_experiment_assistant_contract \
  test_experiment_assistant_datasets \
  test_experiment_assistant_snapshot \
  test_experiment_assistant_runner >"$output" 2>&1
/bin/cat "$output"
test_count=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$output" | /usr/bin/tail -n 1)
test "$test_count" -ge 45

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_STAGE=fixed_transport\n'
"$python" - "$tools" <<'PY'
import sys
from pathlib import Path

tools = Path(sys.argv[1])
sys.path.insert(0, str(tools))
import experiment_assistant_contract as contract
import experiment_assistant_transport as transport

client = transport.CloudExperimentClient()
assert client.remote_argv == [
    "/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
    "convir-4090",
    "/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python",
    (
        "/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-mcp-main/"
        "experience_docx/tools/experiment_assistant_runner.py"
    ),
    "_remote", "--root",
    "/sda/home/wangyuxin/ConvIR-B/runtime/experiment-assistant-candidate",
    "--dataset-registry",
    "/sda/home/wangyuxin/ConvIR-B/runtime/experiment-assistant-datasets.json",
    "--archive-remote", "git@github.com:onenoober/ConvIR-B.git",
]
assert contract.PUBLIC_TOOL_NAMES == (
    "experiment_start", "experiment_status", "experiment_cancel",
    "experiment_repair", "experiment_get", "experiment_search",
)
print("EXPERIMENT_ASSISTANT_PHASE4_FIXED_TRANSPORT_OK")
PY

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_STAGE=stdio\n'
"$python" - "$python" "$tools/convir_experiment_assistant_mcp.py" <<'PY'
import json
import os
import subprocess
import sys

python, server = sys.argv[1:]
environment = os.environ.copy()
process = subprocess.run(
    [python, server],
    input=(
        json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }, separators=(",", ":")) + "\n" +
        json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        }, separators=(",", ":")) + "\n"
    ),
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=10,
    check=True,
    env=environment,
)
assert process.stderr == ""
responses = [json.loads(line) for line in process.stdout.splitlines()]
assert responses[0]["result"]["serverInfo"]["version"] == "0.4.0-candidate"
assert len(responses[1]["result"]["tools"]) == 6
print("EXPERIMENT_ASSISTANT_PHASE4_STDIO_OK")
PY

printf 'EXPERIMENT_ASSISTANT_PHASE4_BRIDGE_CLOUD_OK candidate=%s tests=%s gpu_access=0 real_dataset_access=0 real_protected_data_access=0 real_experiment_launches=0 project_github_main_writes=0\n' \
  "$candidate" "$test_count"
