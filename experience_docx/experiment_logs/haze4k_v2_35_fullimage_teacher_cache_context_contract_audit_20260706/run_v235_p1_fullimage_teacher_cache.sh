#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
REMOTE_ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit"
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706"
OUT_ROOT="/sda/home/wangyuxin/ConvIR-B/runtime_outputs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
LOG="$EVID/runtime_logs/v235_p1_fullimage_teacher_cache.log"
STATUS="$EVID/status.txt"
mkdir -p "$EVID/runtime_logs" "$OUT_ROOT/fullimage_teacher_cache"
cd "$REMOTE_ROOT"
echo "v235_p1_start $(date --iso-8601=seconds) locked=untouched cuda_visible=${CUDA_VISIBLE_DEVICES}" | tee -a "$STATUS"
set +e
"$PY" -u experience_docx/tools/run_haze4k_v235_context_contract_audit.py p1-cache \
  --out-dir "$EVID" \
  --cache-root "$OUT_ROOT/fullimage_teacher_cache" \
  --limit 600 \
  --write-cache \
  --device cuda 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "v235_p1_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then echo V235_P1_COMMAND_OK; else echo V235_P1_COMMAND_FAILED; fi
exit "$rc"
