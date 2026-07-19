#!/usr/bin/env bash
set -euo pipefail

output="/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r6_r5_decision_component_attribution_20260719/r6-a0-r5-decision-attribution-r1"
identity="${output}/control/lifecycle_identity.json"
log="${output}/runtime.log"

if [[ ! -f "${identity}" || ! -f "${log}" ]]; then
  printf 'R6_A0_R1_FAILURE_INSPECTION_FAIL missing_output_artifact\n' >&2
  exit 2
fi

printf 'identity_sha256=%s\n' "$(/usr/bin/sha256sum "${identity}" | /usr/bin/awk '{print $1}')"
printf 'runtime_log_bytes=%s\n' "$(/usr/bin/stat --format='%s' "${log}")"
/usr/bin/tail -n 120 "${log}"
printf 'R6_A0_R1_FAILURE_INSPECTION_OK\n'
