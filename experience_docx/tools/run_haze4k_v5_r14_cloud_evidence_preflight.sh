#!/usr/bin/env bash
set -euo pipefail

ROOT=/sda/home/wangyuxin/ConvIR-B
R3=${ROOT}/runs/haze4k_v5_r3_proposal_first_acv_20260717/r3-a0-proposal-r4
R5=${ROOT}/runs/haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719/r5-a0-spatial-response-screen-r2
R10=${ROOT}/runs/haze4k_v5_r10_fixed_region_action_feasibility_20260719/r10-a0-fixed-region-feasibility-r1
R11=${ROOT}/runs/haze4k_v5_r11_regional_action_observability_20260719/r11-a0-regional-observability-r1
R12=${ROOT}/runs/haze4k_v5_r12_action_conditioned_downside_observability_20260719/r12-a0-action-downside-r2
R13=${ROOT}/runs/haze4k_v5_r13_image_relative_context_observability_20260719/r13-a0-image-relative-context-r1

inspect_file() {
  local label=$1
  local path=$2
  if [[ ! -f "${path}" ]]; then
    printf 'AUDIT_FILE_MISSING\t%s\t%s\n' "${label}" "${path}"
    return 0
  fi
  local sha bytes lines header
  sha=$(/usr/bin/sha256sum "${path}")
  sha=${sha%% *}
  bytes=$(/usr/bin/stat --format='%s' "${path}")
  lines=$(/usr/bin/wc -l < "${path}")
  header=$(/usr/bin/head -n 1 "${path}" | /usr/bin/tr '\t' ' ' | /usr/bin/cut -c 1-2000)
  printf 'AUDIT_FILE\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${sha}" "${bytes}" "${lines}" "${header}" "${path}"
}

inspect_file r3_cache_manifest "${R3}/workload/a0_cache_manifest.json"
inspect_file r3_raw_manifest "${R3}/workload/a0_cache_units_cloud_only.jsonl"
inspect_file r5_candidate_scores "${R5}/workload/r5_a0_candidate_scores_cloud_only.csv"
inspect_file r5_policy_rows "${R5}/workload/r5_a0_policy_rows_cloud_only.csv"
inspect_file r5_seed_rows "${R5}/workload/r5_a0_per_seed_predictions_cloud_only.csv"
inspect_file r10_region_rows "${R10}/workload/r10_a0_per_image_region_rows_cloud_only.csv"
inspect_file r11_tile_rows "${R11}/workload/r11_a0_tile_predictions_cloud_only.csv"
inspect_file r11_policy_rows "${R11}/workload/r11_a0_per_image_policy_rows_cloud_only.csv"
inspect_file r12_risk_rows "${R12}/workload/r12_a0_oof_risk_scores_cloud_only.csv"
inspect_file r13_row_predictions "${R13}/workload/r13_a0_row_predictions_cloud_only.csv"
inspect_file r13_action_maps "${R13}/workload/r13_a0_action_maps_cloud_only.csv"

inspect_file r5_closeout "${R5}/r5_a0_spatial_response_sufficiency_closeout.json"
inspect_file r10_closeout "${R10}/r10_a0_fixed_region_action_feasibility_closeout.json"
inspect_file r11_closeout "${R11}/r11_a0_regional_action_observability_closeout.json"
inspect_file r12_closeout "${R12}/r12_a0_action_conditioned_downside_observability_closeout.json"
inspect_file r13_closeout "${R13}/r13_a0_image_relative_context_observability_closeout.json"
inspect_file r5_workload_closeout "${R5}/workload/r5_a0_spatial_response_sufficiency_closeout.json"
inspect_file r10_workload_closeout "${R10}/workload/r10_a0_fixed_region_action_feasibility_closeout.json"
inspect_file r11_workload_closeout "${R11}/workload/r11_a0_regional_action_observability_closeout.json"
inspect_file r12_workload_closeout "${R12}/workload/r12_a0_action_conditioned_downside_observability_closeout.json"
inspect_file r13_workload_closeout "${R13}/workload/r13_a0_image_relative_context_observability_closeout.json"

for run in "${R5}" "${R10}" "${R11}" "${R12}" "${R13}"; do
  inspect_file run_status "${run}/status.txt"
  printf 'AUDIT_STATUS_TAIL\t%s\n' "${run}"
  /usr/bin/tail -n 4 "${run}/status.txt"
done

R14_ROOT=${ROOT}/runs/haze4k_v5_r14_cross_route_cloud_evidence_audit_20260720
R14_OUTPUT=${R14_ROOT}/r14-cross-route-audit-r1
inspect_file r14_status "${R14_OUTPUT}/status.txt"
inspect_file r14_launcher "${R14_ROOT}/r14-cross-route-audit-r1.launcher.log"
inspect_file r14_closeout "${R14_OUTPUT}/cloud_audit_closeout.json"
if [[ -f "${R14_OUTPUT}/status.txt" ]]; then
  printf 'AUDIT_R14_STATUS_TAIL\n'
  /usr/bin/tail -n 8 "${R14_OUTPUT}/status.txt"
fi
if [[ -f "${R14_ROOT}/r14-cross-route-audit-r1.launcher.log" ]]; then
  printf 'AUDIT_R14_LAUNCHER_TAIL\n'
  /usr/bin/tail -n 20 "${R14_ROOT}/r14-cross-route-audit-r1.launcher.log"
fi
R14_REPO=${ROOT}/repos/ConvIR-B-r14-cross-route-audit-ca93bbb0e
if [[ -d "${R14_REPO}/.git" ]]; then
  printf 'AUDIT_R14_SOURCE_HEAD\t%s\n' "$(/usr/bin/git -C "${R14_REPO}" rev-parse HEAD)"
fi
R14_SYNC_REPO=${ROOT}/repos/ConvIR-B-r14-evidence-sync-r2-2667d59ed
if [[ -d "${R14_SYNC_REPO}/.git" ]]; then
  printf 'AUDIT_R14_SYNC_STAGED_BEGIN\n'
  /usr/bin/git -C "${R14_SYNC_REPO}" diff --cached --name-only
  printf 'AUDIT_R14_SYNC_STAGED_END\n'
fi

printf 'R14_CLOUD_EVIDENCE_PREFLIGHT_OK\n'
