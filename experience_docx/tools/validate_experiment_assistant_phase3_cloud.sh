#!/usr/bin/env bash
set -euo pipefail

on_error() {
  rc=$?
  if [[ -n ${output:-} && -f $output ]]; then
    /bin/cat "$output" >&2
  fi
  printf 'EXPERIMENT_ASSISTANT_PHASE3_CLOUD_FAILED line=%s command=%q rc=%s\n' \
    "$1" "$2" "$rc" >&2
  exit "$rc"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

branch=codex/experiment-assistant-slim-v1
base=43894c4a139aad07da0d65a1fc9835c3b31799a6
github=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
work=$(/usr/bin/mktemp -d /tmp/experiment-assistant-phase3.XXXXXX)

cleanup() {
  case "$work" in
    /tmp/experiment-assistant-phase3.*)
      /bin/rm -rf -- "$work"
      ;;
    *)
      printf 'refusing unsafe temporary cleanup: %s\n' "$work" >&2
      ;;
  esac
}
trap cleanup EXIT

printf 'EXPERIMENT_ASSISTANT_PHASE3_STAGE=checkout\n'
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

printf 'EXPERIMENT_ASSISTANT_PHASE3_STAGE=compile\n'
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

printf 'EXPERIMENT_ASSISTANT_PHASE3_STAGE=tests\n'
output=$work/unittest.txt
"$python" -m unittest -v \
  test_experiment_assistant_contract \
  test_experiment_assistant_datasets \
  test_experiment_assistant_snapshot \
  test_experiment_assistant_runner >"$output" 2>&1
/bin/cat "$output"
test_count=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$output" | /usr/bin/tail -n 1)
test "$test_count" -ge 38

printf 'EXPERIMENT_ASSISTANT_PHASE3_STAGE=stdio\n'
"$python" - "$python" "$tools/convir_experiment_assistant_mcp.py" "$work" <<'PY'
import json
import os
import select
import subprocess
import sys
from pathlib import Path

python, server, raw_root = sys.argv[1:]
root = Path(raw_root)
training = root / "activation-training"
sealed = root / "activation-sealed"
training.mkdir()
sealed.mkdir()
registry = root / "activation-registry.json"
registry.write_text(json.dumps({
    "schema_version": 1,
    "datasets": [
        {
            "id": "activation_train", "role": "training", "path": str(training),
            "identity_sha256": "1" * 64, "protected": False,
        },
        {
            "id": "activation_sealed", "role": "locked_test", "path": str(sealed),
            "identity_sha256": "2" * 64, "protected": True,
        },
    ],
}) + "\n", encoding="utf-8")

seed = root / "activation-seed"
remote = root / "activation-remote.git"
seed.mkdir()
subprocess.run(["/usr/bin/git", "-C", str(seed), "init", "-q"], check=True)
subprocess.run(["/usr/bin/git", "-C", str(seed), "config", "user.name", "Activation"], check=True)
subprocess.run([
    "/usr/bin/git", "-C", str(seed), "config", "user.email", "activation@example.invalid",
], check=True)
(seed / "README.md").write_text("# Activation\n", encoding="utf-8")
subprocess.run(["/usr/bin/git", "-C", str(seed), "add", "README.md"], check=True)
subprocess.run(["/usr/bin/git", "-C", str(seed), "commit", "-qm", "seed"], check=True)
subprocess.run(["/usr/bin/git", "-C", str(seed), "branch", "-M", "main"], check=True)
subprocess.run(["/usr/bin/git", "init", "--bare", "-q", str(remote)], check=True)
subprocess.run(["/usr/bin/git", "-C", str(seed), "remote", "add", "origin", str(remote)], check=True)
subprocess.run(["/usr/bin/git", "-C", str(seed), "push", "-q", "origin", "main"], check=True)
subprocess.run([
    "/usr/bin/git", "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main",
], check=True)

environment = os.environ.copy()
environment.update({
    "CONVIR_EXPERIMENT_ASSISTANT_ROOT": str(root / "activation-state"),
    "CONVIR_EXPERIMENT_ASSISTANT_RUNTIME": "cloud-candidate",
    "CONVIR_EXPERIMENT_ASSISTANT_TEST_MODE": "1",
    "CONVIR_EXPERIMENT_ASSISTANT_LOCAL_TEST_MODE": "1",
    "CONVIR_EXPERIMENT_DATASET_REGISTRY": str(registry),
    "CONVIR_EXPERIMENT_ARCHIVE_ENABLED": "1",
    "CONVIR_EXPERIMENT_ARCHIVE_REMOTE": str(remote),
})
process = subprocess.Popen(
    [python, server],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=environment,
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

assert initialized["result"]["serverInfo"]["version"] == "0.4.0-candidate"
names = tuple(item["name"] for item in listed["result"]["tools"])
assert names == (
    "experiment_start", "experiment_status", "experiment_cancel",
    "experiment_repair", "experiment_get", "experiment_search",
), names
assert missing["result"]["isError"] is True
capabilities = set(missing["result"]["structuredContent"]["capabilities"])
assert {
    "automatic_result_archive", "dataset_registry_resolution",
    "explicit_protected_data_access",
} <= capabilities
experiments = root / "activation-state" / "experiments"
assert list(experiments.iterdir()) == []

process.stdin.close()
assert process.wait(timeout=10) == 0
assert process.stderr.read() == ""
print("EXPERIMENT_ASSISTANT_PHASE3_STDIO_OK")
PY

printf 'EXPERIMENT_ASSISTANT_PHASE3_CLOUD_OK candidate=%s tests=%s gpu_access=0 real_dataset_access=0 real_protected_data_access=0 real_experiment_launches=0 project_github_main_writes=0\n' \
  "$candidate" "$test_count"
