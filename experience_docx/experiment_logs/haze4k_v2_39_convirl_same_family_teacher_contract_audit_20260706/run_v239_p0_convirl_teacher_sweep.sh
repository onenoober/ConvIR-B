#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-39-convirl-same-family-teacher-contract-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
SOURCE_P0="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-37-tail-safe-same-context-wdmamba-eligibility-preservation/experience_docx/experiment_logs/haze4k_v2_37_tail_safe_same_context_wdmamba_eligibility_preservation_20260706/v237_p0_alpha_safety_sweep_per_image.csv"
CACHE_ROOT="/sda/home/wangyuxin/ConvIR-B/runtime_outputs/haze4k_v2_39_convirl_same_family_teacher_contract_audit_20260706/fullimage_convirl_cache"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs" "$CACHE_ROOT"
cd "$REMOTE_ROOT"

echo "v239_p0_preflight $(date --iso-8601=seconds) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) locked=untouched bridge=blocked generator=blocked" | tee -a "$STATUS"
"$PY" -m py_compile experience_docx/tools/run_haze4k_v239_convirl_teacher_audit.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
echo "v239_p0_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v239_convirl_teacher_audit.py \
  --out-dir "$EVID" \
  --source-p0-csv "$SOURCE_P0" \
  --cache-root "$CACHE_ROOT" \
  --device cuda \
  2>&1 | tee "$EVID/runtime_logs/v239_p0_convirl_teacher_sweep.log"
rc=${PIPESTATUS[0]}
set -e
echo "v239_p0_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V239_P0_COMMAND_OK; else echo V239_P0_COMMAND_FAILED; fi
exit "$rc"
