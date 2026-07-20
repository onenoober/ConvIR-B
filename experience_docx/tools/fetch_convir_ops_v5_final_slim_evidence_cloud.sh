#!/usr/bin/env bash
set -euo pipefail

candidate=31cc862f5107106dad8de266299b1bfea0b7a376
root=/sda/home/wangyuxin/ConvIR-B/runs/convir_ops_v5_final_slim_acceptance_20260720
acceptance="$root/acceptance-$candidate.json"
compatibility="$root/r16-compatibility-$candidate.json"
test -f "$acceptance"
test -f "$compatibility"
printf 'FINAL_SLIM_ACCEPTANCE_SHA256=%s\n' "$(sha256sum "$acceptance" | cut -d' ' -f1)"
printf 'FINAL_SLIM_COMPATIBILITY_SHA256=%s\n' "$(sha256sum "$compatibility" | cut -d' ' -f1)"
printf 'FINAL_SLIM_ACCEPTANCE_BEGIN\n'
cat "$acceptance"
printf '\nFINAL_SLIM_ACCEPTANCE_END\n'
printf 'FINAL_SLIM_COMPATIBILITY_BEGIN\n'
cat "$compatibility"
printf '\nFINAL_SLIM_COMPATIBILITY_END\n'
printf 'CONVIR_OPS_V5_FINAL_SLIM_EVIDENCE_FETCH_OK candidate=%s\n' "$candidate"
