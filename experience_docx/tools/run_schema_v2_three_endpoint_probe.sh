#!/usr/bin/env bash
set -Eeuo pipefail

ROUTE_ID="schema_v2_three_endpoint_dispatch_validation_20260714"
BRANCH="codex/model-routing-unknown-host-total-token-20260714"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
EVIDENCE_DIR="${REMOTE_REPO}/experience_docx/experiment_logs/${ROUTE_ID}"
STATUS_PATH="${RUN_ROOT}/status.txt"
CLOSEOUT_PATH="${EVIDENCE_DIR}/schema_v2_three_endpoint_closeout.json"
PROBE_PATH="${EVIDENCE_DIR}/schema_v2_three_endpoint_probe.json"
AUTH_PATH="${EVIDENCE_DIR}/schema_v2_launch_authorization.json"
EXPECTED_AUTH_STATE="CLOUD_ENDPOINT_READY"
EXPECTED_AUTH_DECISION="RUN_SCHEMA_V2_THREE_ENDPOINT_PROBE"
EXPECTED_AUTHORIZES="SCHEMA_V2_CLOUD_PROBE_ONLY"

mkdir -p "${RUN_ROOT}" "${EVIDENCE_DIR}"
exec > >(tee -a "${RUN_ROOT}/stdout.log") 2> >(tee -a "${RUN_ROOT}/stderr.log" >&2)

write_failure() {
  local exit_code=$?
  trap - ERR
  STATUS_PATH="${STATUS_PATH}" CLOSEOUT_PATH="${CLOSEOUT_PATH}" ROUTE_ID="${ROUTE_ID}" EXIT_CODE="${exit_code}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
closeout_path = Path(os.environ["CLOSEOUT_PATH"])
payload = {
    "route_id": os.environ["ROUTE_ID"],
    "state": "FAILED_VALIDATION_COMMAND",
    "decision": "SCHEMA_V2_THREE_ENDPOINT_VALIDATION_FAILED",
    "authorizes": "NONE",
    "failure_class": "VALIDATION_RUNTIME",
    "exit_code": int(os.environ["EXIT_CODE"]),
}
status_path.write_text("FAILED_VALIDATION_COMMAND\n", encoding="utf-8")
closeout_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  exit "${exit_code}"
}
trap write_failure ERR

printf 'state=RUNNING\nroute_id=%s\n' "${ROUTE_ID}" > "${STATUS_PATH}"

test "${MODE}" = "schema-v2-three-end-validation"
test -x "${PY}"
test -f "${AUTH_PATH}"
test "$(git -C "${REMOTE_REPO}" branch --show-current)" = "${BRANCH}"
LOCAL_HEAD="$(git -C "${REMOTE_REPO}" rev-parse HEAD)"
test "${LOCAL_HEAD}" = "${EXPECTED_ROUTE_COMMIT}"

REMOTE_LINE="$(git -C "${REMOTE_REPO}" ls-remote github "refs/heads/${BRANCH}")"
REMOTE_HEAD="$(printf '%s\n' "${REMOTE_LINE}" | awk 'NR == 1 {print $1}')"
test "${REMOTE_HEAD}" = "${EXPECTED_ROUTE_COMMIT}"

AUTH_PATH="${AUTH_PATH}" ROUTE_ID="${ROUTE_ID}" EXPECTED_AUTH_STATE="${EXPECTED_AUTH_STATE}" EXPECTED_AUTH_DECISION="${EXPECTED_AUTH_DECISION}" EXPECTED_AUTHORIZES="${EXPECTED_AUTHORIZES}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

authorization = json.loads(Path(os.environ["AUTH_PATH"]).read_text(encoding="utf-8"))
expected = {
    "route_id": os.environ["ROUTE_ID"],
    "state": os.environ["EXPECTED_AUTH_STATE"],
    "decision": os.environ["EXPECTED_AUTH_DECISION"],
    "authorizes": os.environ["EXPECTED_AUTHORIZES"],
}
for field, value in expected.items():
    assert authorization[field] == value
assert authorization["schema_version"] == 2
assert authorization["gpu_required"] is False
assert authorization["locked_test_authorized"] is False
PY

REMOTE_REPO="${REMOTE_REPO}" EXPECTED_ROUTE_COMMIT="${EXPECTED_ROUTE_COMMIT}" REMOTE_HEAD="${REMOTE_HEAD}" PROBE_PATH="${PROBE_PATH}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REMOTE_REPO"])
schema = json.loads((repo / "experience_docx/tools/agent_model_dispatch_request.schema.json").read_text(encoding="utf-8"))
dispatcher = (repo / "experience_docx/tools/dispatch_agent_task.ps1").read_text(encoding="utf-8")

assert schema["properties"]["schema_version"]["const"] == 2
assert "unknown" in schema["properties"]["source_role"]["enum"]
assert "task_routing" in schema["properties"]["dispatch_reason"]["enum"]
assert "dispatcher_classification" in schema["properties"]["routing_basis"]["enum"]
assert "$sourceRank" not in dispatcher
assert "stage_state -ne \"PASS\"" not in dispatcher
assert "decision -ne \"CONTINUE\"" not in dispatcher

payload = {
    "schema_version": 2,
    "route_commit": os.environ["EXPECTED_ROUTE_COMMIT"],
    "github_branch_commit": os.environ["REMOTE_HEAD"],
    "source_rank_independent": True,
    "route_specific_r1_tuple_supported": True,
}
Path(os.environ["PROBE_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

ROUTE_ID="${ROUTE_ID}" CLOSEOUT_PATH="${CLOSEOUT_PATH}" ROUTE_COMMIT="${EXPECTED_ROUTE_COMMIT}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "route_id": os.environ["ROUTE_ID"],
    "state": "COMPLETED_VALIDATION_PASS",
    "decision": "SCHEMA_V2_THREE_ENDPOINT_PROBE_PASS",
    "authorizes": "SOL_THREE_ENDPOINT_CLOSEOUT_AUDIT_ONLY",
    "failure_class": "NONE",
    "route_commit": os.environ["ROUTE_COMMIT"],
    "schema_version": 2,
    "gpu_used": False,
    "dataset_accessed": False,
    "experiment_modified": False,
}
Path(os.environ["CLOSEOUT_PATH"]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

printf 'state=COMPLETED_VALIDATION_PASS\nroute_id=%s\ncommit=%s\n' "${ROUTE_ID}" "${EXPECTED_ROUTE_COMMIT}" > "${STATUS_PATH}"
echo SCHEMA_V2_THREE_ENDPOINT_CLOUD_OK
