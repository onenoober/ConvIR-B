#!/usr/bin/env bash
set -euo pipefail

route_id="haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
operation_id="R5_A0_SPATIAL_CANDIDATE_RESPONSE_SUFFICIENCY_SCREEN"
run_id="r5-a0-spatial-response-screen-r2"
output_root="/sda/home/wangyuxin/ConvIR-B/runs/${route_id}/${run_id}"
workload_root="${output_root}/workload"
expected_closeout_sha256="e8d6151a9d7fc1198db1db1a0fc44e2de91da0bea0f4a01053c68bf0d5c0e4e7"
closeout_name="r5_a0_spatial_response_sufficiency_closeout.json"

if [[ -f "${workload_root}/${closeout_name}" ]]; then
  closeout_path="${workload_root}/${closeout_name}"
elif [[ -f "${output_root}/${closeout_name}" ]]; then
  closeout_path="${output_root}/${closeout_name}"
else
  printf 'R5_RAW_ARTIFACT_FACTCHECK_FAIL missing_closeout=%s\n' "${closeout_name}" >&2
  exit 2
fi

actual_closeout_sha256="$(/usr/bin/sha256sum "${closeout_path}" | /usr/bin/awk '{print $1}')"
if [[ "${actual_closeout_sha256}" != "${expected_closeout_sha256}" ]]; then
  printf 'R5_RAW_ARTIFACT_FACTCHECK_FAIL closeout_sha256=%s expected=%s\n' \
    "${actual_closeout_sha256}" "${expected_closeout_sha256}" >&2
  exit 3
fi

printf 'route_id=%s\n' "${route_id}"
printf 'operation_id=%s\n' "${operation_id}"
printf 'run_id=%s\n' "${run_id}"
printf 'output_root=%s\n' "${output_root}"
printf 'closeout_path=%s\n' "${closeout_path}"
printf 'closeout_sha256=%s\n' "${actual_closeout_sha256}"

files=(
  "r5_a0_per_seed_predictions_cloud_only.csv"
  "r5_a0_candidate_scores_cloud_only.csv"
  "r5_a0_policy_rows_cloud_only.csv"
)

for name in "${files[@]}"; do
  path="${workload_root}/${name}"
  if [[ ! -f "${path}" ]]; then
    printf 'R5_RAW_ARTIFACT_FACTCHECK_FAIL missing_file=%s\n' "${path}" >&2
    exit 4
  fi
  bytes="$(/usr/bin/stat --format='%s' "${path}")"
  sha256="$(/usr/bin/sha256sum "${path}" | /usr/bin/awk '{print $1}')"
  row_count="$(/usr/bin/awk 'END { print (NR > 0 ? NR - 1 : -1) }' "${path}")"
  IFS= read -r header < "${path}"
  header="${header%$'\r'}"
  printf 'artifact_name=%s\n' "${name}"
  printf 'artifact_path=%s\n' "${path}"
  printf 'artifact_bytes=%s\n' "${bytes}"
  printf 'artifact_sha256=%s\n' "${sha256}"
  printf 'artifact_row_count=%s\n' "${row_count}"
  printf 'artifact_header=%s\n' "${header}"
done

printf 'R5_RAW_ARTIFACT_FACTCHECK_OK\n'
