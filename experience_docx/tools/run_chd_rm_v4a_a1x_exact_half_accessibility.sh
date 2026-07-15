#!/usr/bin/env bash
set -euo pipefail
REMOTE_REPO="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v4a-a1x-exact-half-accessibility-20260715"
RUN_ROOT="/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
EXPLICIT_CLOUD_PYTHON="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
ROUTE_ID="haze4k_v5_chd_rm_v4a_a1x_exact_half_deployable_accessibility_20260715"
ROUTE_COMMIT="${ROUTE_COMMIT:?ROUTE_COMMIT is required}"; RUN_ID="${RUN_ID:?RUN_ID is required}"; AUTHORIZATION_JSON="${AUTHORIZATION_JSON:?AUTHORIZATION_JSON is required}"; DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"; OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:?OFFICIAL_CHECKPOINT is required}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"; STATUS_PATH="${OUTPUT_DIR}/status.jsonl"; HEARTBEAT_PATH="${OUTPUT_DIR}/heartbeat.json"; RUNTIME_LOG_PATH="${OUTPUT_DIR}/runtime.log"; LEARNED_STATE_MANIFEST_PATH="${OUTPUT_DIR}/learned_state_manifest.json"; CLOSEOUT_PATH="${OUTPUT_DIR}/v4a_a1x_s0_closeout.json"
ENTRYPOINT="${REMOTE_REPO}/experience_docx/tools/chd_rm_v4a_a1x_exact_half_accessibility.py"
if [ "${1:-}" != "s0" ] || [ -e "${OUTPUT_DIR}" ] || [ ! -f "${AUTHORIZATION_JSON}" ] || [ ! -f "${OFFICIAL_CHECKPOINT}" ]; then exit 2; fi
mkdir -p "${OUTPUT_DIR}"
set +e
"${EXPLICIT_CLOUD_PYTHON}" "${ENTRYPOINT}" --stage s0 --authorization-json "${AUTHORIZATION_JSON}" --route-commit "${ROUTE_COMMIT}" --run-id "${RUN_ID}" --official-checkpoint "${OFFICIAL_CHECKPOINT}" --official-checkpoint-sha256 "6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088" --hazy-root "${DATA_ROOT}/hazy" --frozen-base-root "${DATA_ROOT}/frozen_base" --old-0125-root "${DATA_ROOT}/old_0125" --old-025-root "${DATA_ROOT}/old_025" --current-delta-root "${DATA_ROOT}/current_delta" --target-delta-root "${DATA_ROOT}/target_delta" --status-json "${STATUS_PATH}" --heartbeat-json "${HEARTBEAT_PATH}" --learned-state-manifest-json "${LEARNED_STATE_MANIFEST_PATH}" --closeout-json "${CLOSEOUT_PATH}" 2>&1 | tee "${RUNTIME_LOG_PATH}"
exit_code="${PIPESTATUS[0]}"
set -e
if [ ! -f "${CLOSEOUT_PATH}" ]; then exit 2; fi
if [ "${exit_code}" -eq 0 ]; then printf '%s\n' "A1X_S0_OK"; else printf '%s\n' "A1X_S0_FAILED" >&2; fi
exit "${exit_code}"
