#!/usr/bin/env bash
set -euo pipefail

runtime_root=/sda/home/wangyuxin/ConvIR-B/runtime
pattern='/sda/home/wangyuxin/ConvIR-B/runtime/convir-evidence-review-phase4b\.'
test -d "$runtime_root"

mapfile -t directories < <(
  find "$runtime_root" -mindepth 1 -maxdepth 1 -type d \
    -name 'convir-evidence-review-phase4b.*' -print | sort
)
mapfile -t processes < <(pgrep -af "$pattern" || true)

printf 'CONVIR_EVIDENCE_REVIEW_PHASE4B_INSPECTION directories=%s matching_processes=%s\n' \
  "${#directories[@]}" "${#processes[@]}"
printf 'directory=%s\n' "${directories[@]}"
printf 'process=%s\n' "${processes[@]}"
