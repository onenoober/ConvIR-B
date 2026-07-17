#!/usr/bin/env bash
set -euo pipefail

branch=codex/route-ready-runner-v1-20260717
repo_url=git@github.com:onenoober/ConvIR-B.git
seed=/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-official-arch-anchor
python=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
candidate=$(/usr/bin/git ls-remote "$repo_url" "refs/heads/$branch" | /usr/bin/awk 'NR==1 {print $1}')
[[ $candidate =~ ^[0-9a-f]{40}$ ]]
run_root="/tmp/convir-ops-4-2-${candidate:0:12}"
checkout="$run_root/repository"
status="$run_root/status.txt"
stdout="$run_root/stdout.log"
stderr="$run_root/stderr.log"

case "$run_root" in /tmp/convir-ops-4-2-[0-9a-f]*) ;; *) exit 91 ;; esac
if [[ -e $run_root ]]; then
  echo "CONVIR_OPS_4_2_VALIDATION_FAILED existing_run_root=$run_root" >&2
  exit 1
fi
mkdir -p "$run_root"
printf 'state=RUNNING\ncandidate=%s\nmodel_calls=0\n' "$candidate" > "$status"

cleanup() {
  rc=$?
  case "$checkout" in /tmp/convir-ops-4-2-[0-9a-f]*/repository) rm -rf -- "$checkout" ;; *) rc=92 ;; esac
  if [[ $rc -ne 0 ]]; then
    printf 'state=FAILED\ncandidate=%s\nmodel_calls=0\nexit_code=%s\n' "$candidate" "$rc" > "$status"
  fi
  exit "$rc"
}
trap cleanup EXIT

test -d "$seed/.git"
/usr/bin/git clone --quiet --shared --no-checkout "$seed" "$checkout"
/usr/bin/git -C "$checkout" remote rename origin seed
/usr/bin/git -C "$checkout" remote add github "$repo_url"
/usr/bin/git -C "$checkout" fetch --quiet --no-tags --depth=1 github \
  "+refs/heads/$branch:refs/remotes/github/$branch"
/usr/bin/git -C "$checkout" checkout --quiet --detach "$candidate"
test "$(/usr/bin/git -C "$checkout" rev-parse HEAD)" = "$candidate"

"$python" -m py_compile \
  "$checkout/experience_docx/tools/convir_ops_mcp.py" \
  "$checkout/experience_docx/tools/route_lifecycle.py" \
  "$checkout/experience_docx/tools/route_program_api.py" \
  "$checkout/experience_docx/tools/route_runtime_contract.py" \
  "$checkout/experience_docx/tools/validate_route_ready.py"

set +e
"$python" -m unittest discover -s "$checkout/experience_docx/tools/tests" -p 'test_*.py' \
  >"$stdout" 2>"$stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  /usr/bin/tail -n 120 "$stdout" >&2 || true
  /usr/bin/tail -n 120 "$stderr" >&2 || true
  exit "$rc"
fi
tests=$(/usr/bin/sed -nE 's/^Ran ([0-9]+) tests?.*/\1/p' "$stderr" | /usr/bin/tail -n 1)
[[ $tests =~ ^[0-9]+$ ]]

"$python" - "$checkout/experience_docx/tools/convir_ops_mcp.py" <<'PY'
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("convir_ops", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SERVER_VERSION == "4.2.0"
assert module.SCHEMA_VERSION == 4
assert len(module.TOOLS) == 6
assert "clone --quiet --shared --no-checkout" in module.atomic_start_body({
    "remote_repo": "/remote/repo", "branch": "codex/test",
    "route_branch_commit": "a" * 40, "runner_relpath": "experience_docx/tools/run_test.sh",
    "runner_sha256": "b" * 64, "run_root": "/runs/test",
    "output_path": "/runs/test/run", "closeout_path": "/remote/repo/closeout.json",
    "session": "convir-test", "workspace_policy": "fresh_route",
    "output_policy": "new", "mode": "test", "output_id": "run",
}, None)
print("CONVIR_OPS_4_2_CONTRACT_OK tools=6 model_calls=0")
PY

/usr/bin/git -C "$checkout" diff --check
test -z "$(/usr/bin/git -C "$checkout" status --porcelain)"
printf 'state=PASS\ncandidate=%s\nmodel_calls=0\ntests=%s\ntools=6\n' \
  "$candidate" "$tests" > "$status"
echo "CONVIR_OPS_4_2_VALIDATION_OK candidate=$candidate tests=$tests tools=6 model_calls=0"
