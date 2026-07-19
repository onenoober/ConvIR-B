#!/usr/bin/env bash
set -euo pipefail

route_id="haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719"
operation_id="R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN"
run_id="r5-a0-spatial-response-screen-r2"
route_commit="7e75eed504b2ead65a1971ec250dc7f59a79574d"
runner_sha256="336c7e1beccb793229beb533ba12367261e702866497c388ee2a4fa88d12718b"
output_root="/sda/home/wangyuxin/ConvIR-B/runs/${route_id}/${run_id}"
workload_root="${output_root}/workload"
identity_path="${output_root}/control/lifecycle_identity.json"
cloud_python="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"

if [[ ! -f "${identity_path}" ]]; then
  printf 'R5_RAW_ARTIFACT_FACTCHECK_FAIL missing_identity=%s\n' "${identity_path}" >&2
  exit 2
fi

if ! identity_values="$("${cloud_python}" -c 'import json, sys
p, route, operation, run, commit, runner = sys.argv[1:]
with open(p, encoding="utf-8") as stream:
    value = json.load(stream)
expected = {"schema_version": 1, "route_id": route, "operation_id": operation, "run_id": run, "route_commit": commit, "runner_sha256": runner}
if value != expected:
    raise SystemExit(5)
print("\t".join((route, operation, run, commit, runner)))' \
    "${identity_path}" "${route_id}" "${operation_id}" "${run_id}" \
    "${route_commit}" "${runner_sha256}")"; then
  printf 'R5_RAW_ARTIFACT_FACTCHECK_FAIL identity_mismatch=%s\n' "${identity_path}" >&2
  exit 3
fi
IFS=$'\t' read -r observed_route observed_operation observed_run observed_commit observed_runner \
  <<< "${identity_values}"

printf 'route_id=%s\n' "${observed_route}"
printf 'operation_id=%s\n' "${observed_operation}"
printf 'run_id=%s\n' "${observed_run}"
printf 'route_commit=%s\n' "${observed_commit}"
printf 'runner_sha256=%s\n' "${observed_runner}"
printf 'output_root=%s\n' "${output_root}"
printf 'identity_path=%s\n' "${identity_path}"
printf 'identity_sha256=%s\n' "$(/usr/bin/sha256sum "${identity_path}" | /usr/bin/awk '{print $1}')"

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
