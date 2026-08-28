#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  if [[ -n ${output:-} && -f $output ]]; then
    /bin/cat "$output" >&2
  fi
  printf 'EXPERIMENT_ASSISTANT_PHASE2_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-assistant-slim-v1
base=2eec1b650afef4c65388ab74c262538d2e7c6915
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/experiment-assistant-phase2.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/experiment-assistant-phase2.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'EXPERIMENT_ASSISTANT_PHASE2_STAGE=checkout\n'
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
  experience_docx/route_operations.json \
  experience_docx/AI_POLICY_SNAPSHOT.json \
  AGENTS.md

tools=$work/repo/experience_docx/tools
tests=$tools/tests
export PYTHONPATH="$tools:$tests"
export CUDA_VISIBLE_DEVICES=""

printf 'EXPERIMENT_ASSISTANT_PHASE2_STAGE=compile\n'
"$python" -m py_compile \
  "$tools/experiment_assistant_contract.py" \
  "$tools/experiment_assistant_snapshot.py" \
  "$tools/experiment_assistant_runner.py" \
  "$tools/convir_experiment_assistant_mcp.py" \
  "$tests/test_experiment_assistant_contract.py" \
  "$tests/test_experiment_assistant_snapshot.py" \
  "$tests/test_experiment_assistant_runner.py"

printf 'EXPERIMENT_ASSISTANT_PHASE2_STAGE=tests\n'
output=$work/unittest.txt
"$python" -m unittest -v \
  test_experiment_assistant_contract \
  test_experiment_assistant_snapshot \
  test_experiment_assistant_runner >"$output" 2>&1
/bin/cat "$output"
test_count=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$output" | /usr/bin/tail -n 1)
test "$test_count" -ge 29

printf 'EXPERIMENT_ASSISTANT_PHASE2_STAGE=stdio\n'
export CONVIR_EXPERIMENT_ASSISTANT_ROOT="$work/activation-state"
export CONVIR_EXPERIMENT_ASSISTANT_RUNTIME=cloud-candidate
"$python" - "$python" "$tools/convir_experiment_assistant_mcp.py" <<'PY'
import json
import os
import select
import subprocess
import sys

python, server = sys.argv[1:]
process = subprocess.Popen(
    [python, server],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=os.environ.copy(),
)

def request(payload):
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], 10)
    assert ready, "stdio response timeout"
    line = process.stdout.readline()
    assert line, "stdio server closed"
    return json.loads(line)

initialized = request({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05"},
})
listed = request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
missing = request({
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {
        "name": "experiment_status",
        "arguments": {"experiment_id": "does-not-exist"},
    },
})

assert initialized["result"]["serverInfo"]["version"] == "0.2.0-candidate"
names = tuple(item["name"] for item in listed["result"]["tools"])
assert names == (
    "experiment_start", "experiment_status", "experiment_cancel",
    "experiment_repair", "experiment_get", "experiment_search",
), names
assert missing["result"]["isError"] is True
experiments = os.path.join(os.environ["CONVIR_EXPERIMENT_ASSISTANT_ROOT"], "experiments")
assert os.listdir(experiments) == [], os.listdir(experiments)

process.stdin.close()
assert process.wait(timeout=10) == 0
assert process.stderr.read() == ""
print("EXPERIMENT_ASSISTANT_PHASE2_STDIO_OK")
PY

printf 'EXPERIMENT_ASSISTANT_PHASE2_CLOUD_OK candidate=%s tests=%s gpu_access=0 dataset_access=0 protected_data_access=0 real_experiment_launches=0 github_archive_writes=0\n' \
  "$candidate" "$test_count"
