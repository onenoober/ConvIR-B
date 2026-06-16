#!/usr/bin/env bash
set -euo pipefail

ROOT="/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c9-fixed-wdmamba-router"
PY="/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python"
C8="$ROOT/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615"
EVID="$ROOT/experience_docx/experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615"
TOOL="$ROOT/experience_docx/tools/analyze_haze4k_v22_c9_fixed_router.py"
STATUS="$EVID/status_c9.txt"
LOG="$EVID/runtime_logs/v22_c9_table_analysis.log"

mkdir -p "$EVID/runtime_logs"
echo "v22_c9_table_analysis_start $(date -Is) locked=untouched source=c8_train_derived_tables" | tee -a "$STATUS"
"$PY" "$TOOL" --c8-root "$C8" --out-dir "$EVID" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "v22_c9_table_analysis_done rc=$rc $(date -Is)" | tee -a "$STATUS"
exit "$rc"
