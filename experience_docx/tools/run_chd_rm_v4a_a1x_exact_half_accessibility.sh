#!/usr/bin/env bash
set -euo pipefail
REMOTE_REPO="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1x-exact-half-accessibility-20260715"
RUN_ROOT="/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
EXPLICIT_CLOUD_PYTHON="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
ROUTE_ID="haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
ROUTE_COMMIT="${ROUTE_COMMIT:?ROUTE_COMMIT is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
AUTHORIZATION_JSON="${AUTHORIZATION_JSON:?AUTHORIZATION_JSON is required}"
A1X_RUNTIME_ASSET_MANIFEST_JSON="${A1X_RUNTIME_ASSET_MANIFEST_JSON:?A1X_RUNTIME_ASSET_MANIFEST_JSON is required}"
A1X_RUNTIME_ASSET_MANIFEST_SHA256="${A1X_RUNTIME_ASSET_MANIFEST_SHA256:?A1X_RUNTIME_ASSET_MANIFEST_SHA256 is required}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
STATUS_PATH="${OUTPUT_DIR}/status.jsonl"; HEARTBEAT_PATH="${OUTPUT_DIR}/heartbeat.json"; RUNTIME_LOG_PATH="${OUTPUT_DIR}/runtime.log"
LEARNED_STATE_MANIFEST_PATH="${OUTPUT_DIR}/learned_state_manifest.json"; CLOSEOUT_PATH="${OUTPUT_DIR}/v4a_a1x_s0_closeout.json"
ENTRYPOINT="${REMOTE_REPO}/experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"
if [ "${1:-}" != "s0" ] || [ -e "${OUTPUT_DIR}" ] || [ ! -f "${AUTHORIZATION_JSON}" ]; then exit 2; fi
mkdir -p "${OUTPUT_DIR}"
set +e
"${EXPLICIT_CLOUD_PYTHON}" "${ENTRYPOINT}" --stage s0 --authorization-json "${AUTHORIZATION_JSON}" --route-commit "${ROUTE_COMMIT}" --run-id "${RUN_ID}" --run-root "${OUTPUT_DIR}" --runtime-asset-manifest-json "${A1X_RUNTIME_ASSET_MANIFEST_JSON}" --runtime-asset-manifest-sha256 "${A1X_RUNTIME_ASSET_MANIFEST_SHA256}" --status-json "${STATUS_PATH}" --heartbeat-json "${HEARTBEAT_PATH}" --learned-state-manifest-json "${LEARNED_STATE_MANIFEST_PATH}" --closeout-json "${CLOSEOUT_PATH}" 2>&1 | tee "${RUNTIME_LOG_PATH}"
exit_code="${PIPESTATUS[0]}"
set -e
if [ ! -f "${CLOSEOUT_PATH}" ]; then exit_code=2; fi
if [ "${exit_code}" -eq 0 ]; then printf '%s\n' "A1X_S0_OK"; else printf '%s\n' "A1X_S0_FAILED" >&2; fi
exit "${exit_code}"
