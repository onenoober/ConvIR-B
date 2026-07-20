#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720/r14-cross-route-audit-r2
FILES=(
  input_identity.json
  raw_contract_checks.json
  official_result_reproduction.json
  a1_regret_attribution.json
  a2_label_region_stability.json
  a3_risk_observability.json
  a4_target_alignment.json
  a5_external_directional_reference.json
  key_raw_findings.json
  cloud_audit_closeout.json
  scientific_conclusion.json
)
for name in "${FILES[@]}"; do
  path=${ROOT}/${name}
  if [[ ! -f "${path}" ]]; then
    printf 'R14_R2_EVIDENCE_MISSING\t%s\n' "${name}" >&2
    exit 2
  fi
  printf 'R14_R2_FILE_BEGIN\t%s\t%s\t%s\n' \
    "${name}" "$(/usr/bin/sha256sum "${path}" | /usr/bin/cut -d ' ' -f 1)" \
    "$(/usr/bin/stat --format='%s' "${path}")"
  /usr/bin/cat "${path}"
  printf 'R14_R2_FILE_END\t%s\n' "${name}"
done
printf 'R14_R2_EVIDENCE_INSPECTION_OK\n'
