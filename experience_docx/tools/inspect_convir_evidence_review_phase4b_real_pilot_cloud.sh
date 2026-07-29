#!/usr/bin/env bash
set -euo pipefail

runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
name='convir-evidence-review-phase4b-real-pilot.*'
pattern='/sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b-real-pilot\.'
test -d "$runtime_root"

mapfile -t directories < <(
  find "$runtime_root" -mindepth 1 -maxdepth 1 -type d -name "$name" -print | sort
)
mapfile -t processes < <(pgrep -af "$pattern" || true)

printf 'CONVIR_EVIDENCE_REVIEW_PHASE4B_REAL_PILOT_INSPECTION directories=%s matching_processes=%s\n' \
  "${#directories[@]}" "${#processes[@]}"
printf 'directory=%s\n' "${directories[@]}"
printf 'process=%s\n' "${processes[@]}"
