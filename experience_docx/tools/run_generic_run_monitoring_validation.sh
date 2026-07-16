#!/usr/bin/env bash
set -euo pipefail

ROUTE_ID=generic_run_monitoring_validation_20260716
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
TELEMETRY="$REMOTE_REPO/experience_docx/tools/run_telemetry.py"
VALIDATOR="$REMOTE_REPO/experience_docx/tools/validate_generic_run_monitoring.py"
EVID_STAGE="$REMOTE_REPO/experience_docx/experiment_logs/$ROUTE_ID"
STATUS_PATH="$OUTPUT_PATH/status.txt"
HEARTBEAT_PATH="$OUTPUT_PATH/heartbeat.json"
RUNTIME_LOG="$OUTPUT_PATH/runtime.log"
CLOSEOUT="$EVID_STAGE/generic_run_monitoring_validation_closeout.json"
SUMMARY="$EVID_STAGE/generic_run_monitoring_validation_summary.json"

: "${REMOTE_REPO:?REMOTE_REPO is required}"
: "${OUTPUT_PATH:?OUTPUT_PATH is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${EXPECTED_ROUTE_COMMIT:?EXPECTED_ROUTE_COMMIT is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 is required}"

mkdir -p "$OUTPUT_PATH" "$EVID_STAGE"
test -x "$PY"
test -f "$TELEMETRY"
test -f "$VALIDATOR"
test "$(git -C "$REMOTE_REPO" rev-parse HEAD)" = "$EXPECTED_ROUTE_COMMIT"
test -z "$(git -C "$REMOTE_REPO" status --porcelain)"
test "$(sha256sum "$REMOTE_REPO/experience_docx/tools/run_generic_run_monitoring_validation.sh" | awk '{print $1}')" = "$RUNNER_SHA256"

write_closeout() {
  local state=$1 decision=$2 authorizes=$3 pass=$4
  "$PY" - "$CLOSEOUT" "$state" "$decision" "$authorizes" "$pass" <<'PY'
import json, os, sys, tempfile, time
from pathlib import Path
path = Path(sys.argv[1])
decision = None if sys.argv[3] == "null" else sys.argv[3]
value = {
    "schema_version": 1,
    "route_id": "generic_run_monitoring_validation_20260716",
    "run_id": os.environ["RUN_ID"],
    "route_commit": os.environ["EXPECTED_ROUTE_COMMIT"],
    "runner_sha256": os.environ["RUNNER_SHA256"],
    "state": sys.argv[2],
    "decision": decision,
    "authorizes": sys.argv[4],
    "validation_pass": sys.argv[5] == "true",
    "model_calls": 0,
    "gpu_used": False,
    "dataset_touched": False,
    "checkpoint_touched": False,
    "canary_touched": False,
    "locked_test_touched": False,
    "timestamp_unix": time.time(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=".closeout.", suffix=".tmp", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(name, path)
PY
}

failure_closeout() {
  local rc=$?
  set +e
  write_closeout FAILED_ENGINEERING null NONE false
  echo "GENERIC_RUN_MONITORING_VALIDATION_FAILED rc=$rc" | tee -a "$STATUS_PATH"
  exit "$rc"
}
trap failure_closeout ERR

"$PY" "$TELEMETRY" sidecar \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase validation \
  --heartbeat "$HEARTBEAT_PATH" --parent-pid "$$" --interval-seconds 1 &

"$PY" "$TELEMETRY" event \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase validation \
  --status "$STATUS_PATH" --event validation_start --completed 0 --total 3

timeout --signal=TERM --kill-after=30s 8m \
  "$PY" "$REMOTE_REPO/experience_docx/tools/tests/test_run_telemetry.py" \
  2>&1 | tee -a "$RUNTIME_LOG"

"$PY" "$TELEMETRY" event \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase validation \
  --status "$STATUS_PATH" --event telemetry_tests_pass --completed 1 --total 3

timeout --signal=TERM --kill-after=30s 8m \
  "$PY" "$REMOTE_REPO/experience_docx/tools/tests/test_convir_ops_mcp.py" \
  2>&1 | tee -a "$RUNTIME_LOG"

"$PY" "$TELEMETRY" event \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase validation \
  --status "$STATUS_PATH" --event control_tests_pass --completed 2 --total 3

"$PY" "$VALIDATOR" --telemetry "$TELEMETRY" --output "$SUMMARY" \
  2>&1 | tee -a "$RUNTIME_LOG"

"$PY" "$TELEMETRY" event \
  --route-id "$ROUTE_ID" --run-id "$RUN_ID" --phase validation \
  --status "$STATUS_PATH" --event validation_pass --completed 3 --total 3

write_closeout COMPLETED_GATE_PASS GENERIC_RUN_MONITORING_VALIDATION_PASS MAIN_INTEGRATION_REVIEW_ONLY true
trap - ERR
echo GENERIC_RUN_MONITORING_VALIDATION_OK | tee -a "$STATUS_PATH"
