#!/bin/bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
PYTHON=$BASE/envs/convir-cu121/bin/python
REMOTE_URL=git@github.com:onenoober/ConvIR-B.git
BRANCH=codex/receipt-bound-operator-control-v1
SEED=$BASE/repos/ConvIR-B-official-arch-anchor
REMOTE_LINE=$(/usr/bin/git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH")
read -r COMMIT REMOTE_REF <<< "$REMOTE_LINE"
test "$REMOTE_REF" = "refs/heads/$BRANCH"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
WORK=$(mktemp -d "$BASE/runs/operator-control-validation.XXXXXXXX")
case "$WORK" in "$BASE/runs/operator-control-validation."*) ;; *) exit 91 ;; esac
REPO=$WORK/repo
TOOLS=$REPO/experience_docx/tools

cleanup() {
  code=$?
  trap - EXIT
  case "$WORK" in "$BASE/runs/operator-control-validation."*) rm -rf -- "$WORK" ;; *) code=91 ;; esac
  exit "$code"
}
trap cleanup EXIT

test -x "$PYTHON"
test -d "$SEED/.git" || test -f "$SEED/HEAD"
/usr/bin/git clone --quiet --no-checkout --reference-if-able "$SEED" "$REMOTE_URL" "$REPO"
/usr/bin/git -C "$REPO" fetch --quiet --no-tags "$REMOTE_URL" \
  "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
/usr/bin/git -C "$REPO" checkout --quiet --detach "$COMMIT"
test "$(/usr/bin/git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain)"

"$PYTHON" -m py_compile \
  "$TOOLS/convir_ops_mcp.py" \
  "$TOOLS/route_lifecycle.py" \
  "$TOOLS/tests/test_convir_ops_mcp.py" \
  "$TOOLS/tests/test_route_lifecycle.py"

PYTHONPATH="$TOOLS" "$PYTHON" -m unittest discover \
  -s "$TOOLS/tests" -p 'test_*.py'

PYTHONPATH="$TOOLS" "$PYTHON" - "$REPO" <<'PY'
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import convir_ops_mcp as ops

source_repo = Path(sys.argv[1])
route_id = "operator-control-e2e-" + uuid.uuid4().hex[:12]
run_id = "operator-cancel-e2e"
remote_repo = Path(ops.derive_remote_repo(route_id, run_id))
run_root = Path(ops.REMOTE_RUNS) / route_id
output = run_root / run_id
runner_rel = "experience_docx/tools/run_route_operation.sh"
runner = remote_repo / runner_rel
lifecycle = remote_repo / "experience_docx/tools/route_lifecycle.py"
closeout = remote_repo / "experience_docx/experiment_logs" / route_id / "e2e_closeout.json"
session = ops.derive_session(route_id, "s0", "0" * 40, run_id)

assert remote_repo.parent == Path(ops.REMOTE_REPOS)
assert run_root.parent == Path(ops.REMOTE_RUNS)
assert not remote_repo.exists()
assert not run_root.exists()

try:
    runner.parent.mkdir(parents=True)
    closeout.parent.mkdir(parents=True)
    output.joinpath("control").mkdir(parents=True)
    runner.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="utf-8")
    lifecycle.write_text(
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, lambda *_: None)\n"
        "while True: time.sleep(0.1)\n",
        encoding="utf-8",
    )
    subprocess.run(["/usr/bin/git", "init", "-q", str(remote_repo)], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(remote_repo), "add", "."], check=True)
    subprocess.run([
        "/usr/bin/git", "-C", str(remote_repo),
        "-c", "user.name=operator-control-e2e",
        "-c", "user.email=operator-control-e2e@invalid",
        "commit", "-qm", "operator control e2e",
    ], check=True)
    commit = subprocess.check_output([
        "/usr/bin/git", "-C", str(remote_repo), "rev-parse", "HEAD",
    ], text=True).strip()
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    identity = {
        "schema_version": 1, "route_id": route_id, "operation_id": "S0",
        "run_id": run_id, "route_commit": commit, "runner_sha256": runner_sha,
    }
    output.joinpath("control/lifecycle_identity.json").write_text(
        json.dumps(identity), encoding="utf-8",
    )
    output.joinpath("status.txt").write_text(
        '{"R3_PROGRESS":{"stage":"result_blind_extract",'
        '"completed_units":2,"total_units":9,"metric":99.9,'
        '"sample_id":"must_not_escape"}}\n',
        encoding="utf-8",
    )
    process_env = os.environ.copy()
    process_env.update({
        "EXPECTED_ROUTE_COMMIT": commit,
        "RUNNER_SHA256": runner_sha,
        "MODE": "s0",
        "REMOTE_REPO": str(remote_repo),
        "RUN_ROOT": str(run_root),
        "OUTPUT_PATH": str(output),
        "RUN_ID": run_id,
        "OUTPUT_ID": run_id,
        "GPU": "",
    })
    subprocess.run([
        ops.REMOTE_TMUX, "new-session", "-d", "-s", session,
        ops.REMOTE_PYTHON, str(lifecycle),
    ], env=process_env, check=True)
    time.sleep(0.5)
    context = {
        "route_id": route_id, "operation_id": "S0", "mode": "s0",
        "output_id": run_id, "remote_repo": str(remote_repo),
        "run_root": str(run_root), "output_path": str(output),
        "closeout_filename": closeout.name, "closeout_path": str(closeout),
        "session": session, "route_branch_commit": commit,
        "runner_relpath": runner_rel, "runner_sha256": runner_sha,
        "allowed_terminal_tuples": [
            {"state": "FAILED_ENGINEERING", "decision": None, "authorizes": "NONE"},
        ],
    }
    prior_grace = ops.OPERATOR_CANCEL_GRACE_SECONDS
    prior_force = ops.OPERATOR_CANCEL_FORCE_SECONDS
    ops.OPERATOR_CANCEL_GRACE_SECONDS = 1
    ops.OPERATOR_CANCEL_FORCE_SECONDS = 1
    try:
        body = ops.operator_cancel_body(context, "1" * 32)
    finally:
        ops.OPERATOR_CANCEL_GRACE_SECONDS = prior_grace
        ops.OPERATOR_CANCEL_FORCE_SECONDS = prior_force
    completed = subprocess.run(
        ["/bin/bash"], input="set -euo pipefail\n" + body + "\n",
        text=True, capture_output=True, timeout=20,
    )
    if completed.returncode:
        raise AssertionError(completed.stdout + "\n" + completed.stderr)
    observed = ops.parse_operator_cancel(completed.stdout)
    parsed = ops.parse_closeout(context, completed.stdout)
    assert observed["termination_mode"] == "forced", observed
    assert parsed["terminal_tuple"] == ops.OPERATOR_CANCEL_TERMINAL, parsed
    cancellation = parsed["operator_cancellation"]
    assert cancellation["completed_units"] == 2, cancellation
    assert cancellation["total_units"] == 9, cancellation
    assert cancellation["stage"] == "result_blind_extract", cancellation
    assert cancellation["scientific_result_interpretable"] is False
    assert "must_not_escape" not in completed.stdout
    assert "99.9" not in completed.stdout
    session_check = subprocess.run(
        [ops.REMOTE_TMUX, "has-session", "-t", session],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert session_check.returncode != 0
finally:
    subprocess.run(
        [ops.REMOTE_TMUX, "kill-session", "-t", session],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if remote_repo.exists():
        assert remote_repo.parent == Path(ops.REMOTE_REPOS)
        shutil.rmtree(remote_repo)
    if run_root.exists():
        assert run_root.parent == Path(ops.REMOTE_RUNS)
        shutil.rmtree(run_root)

print("RECEIPT_BOUND_OPERATOR_CONTROL_E2E_OK")
PY

printf 'state=COMPLETED_GATE_PASS\ncommit=%s\nmarker=RECEIPT_BOUND_OPERATOR_CONTROL_V1_CLOUD_OK\n' "$COMMIT"
