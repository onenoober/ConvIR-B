#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$OUT/status.txt"
LOG="$OUT/v3d_matched_control_compare.log"

mkdir -p "$OUT"
echo "matched_control_compare_start haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/compare_chd_rm_v3d_matched_control.py \
  --d7c_summary "$OUT/v3d_stage1_5epoch_audit_summary.json" \
  --d7c_per_image "$OUT/v3d_stage1_5epoch_val_inner_per_image.csv" \
  --control_summary "$OUT/v3d_fam2modres_control_5epoch_audit_summary.json" \
  --control_per_image "$OUT/v3d_fam2modres_control_5epoch_val_inner_per_image.csv" \
  --output_dir "$OUT" \
  > "$LOG" 2>&1
rc=$?
set -e
echo "matched_control_compare_done rc=$rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3D_MATCHED_CONTROL_COMPARISON_PAUSE_OK" | tee -a "$STATUS"
elif [ "$rc" -eq 2 ]; then
  echo "V3D_MATCHED_CONTROL_COMPARISON_CONTINUE_REQUIRES_DECISION" | tee -a "$STATUS"
  exit 0
else
  echo "V3D_MATCHED_CONTROL_COMPARISON_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
