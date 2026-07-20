#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B
PYTHON=${ROOT}/envs/convir-cu121/bin/python
SOURCE_COMMIT=fe4005f00128346872bccd2107a16973e52a417b
SOURCE_REPO=${ROOT}/repos/ConvIR-B-r14-cross-route-audit-fe4005f00
RUN_ROOT=${ROOT}/runs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720
OUTPUT=${RUN_ROOT}/r14-cross-route-audit-r2
LAUNCH_LOG=${RUN_ROOT}/r14-cross-route-audit-r2.launcher.log

if [[ ! -x "${PYTHON}" ]]; then
  printf 'R14_AUDIT_PYTHON_MISSING\n' >&2
  exit 2
fi
if [[ -e "${SOURCE_REPO}" || -e "${OUTPUT}" ]]; then
  printf 'R14_AUDIT_FRESH_PATH_REQUIRED\n' >&2
  exit 3
fi
/usr/bin/mkdir -p "${ROOT}/repos" "${RUN_ROOT}"
/usr/bin/git clone --no-checkout git@github.com:onenoober/ConvIR-B.git "${SOURCE_REPO}" >"${LAUNCH_LOG}" 2>&1
/usr/bin/git -C "${SOURCE_REPO}" checkout --detach "${SOURCE_COMMIT}" >>"${LAUNCH_LOG}" 2>&1
ACTUAL_COMMIT=$(/usr/bin/git -C "${SOURCE_REPO}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${SOURCE_COMMIT}" ]]; then
  printf 'R14_AUDIT_SOURCE_MISMATCH\n' >&2
  exit 4
fi
"${PYTHON}" "${SOURCE_REPO}/experience_docx/tools/audit_haze4k_v5_r14_cross_route_cloud_evidence.py" \
  --repo-root "${SOURCE_REPO}" \
  --output "${OUTPUT}" \
  --source-commit "${SOURCE_COMMIT}" >>"${LAUNCH_LOG}" 2>&1
/usr/bin/cp "${LAUNCH_LOG}" "${OUTPUT}/runtime.log"
/usr/bin/grep -F 'R14_AUDIT_COMPLETED' "${OUTPUT}/status.txt" >/dev/null
/usr/bin/grep -F 'R14_CROSS_ROUTE_CLOUD_AUDIT_OK' "${OUTPUT}/runtime.log" >/dev/null
printf 'R14_CROSS_ROUTE_CLOUD_AUDIT_R2_OK\n'
