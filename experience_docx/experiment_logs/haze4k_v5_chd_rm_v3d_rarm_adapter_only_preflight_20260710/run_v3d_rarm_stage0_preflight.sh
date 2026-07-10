#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$OUT/status.txt"
LOG="$OUT/v3d_stage0_preflight.log"

mkdir -p "$OUT"
{
  echo "stage0_preflight_start haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)"
  echo "root=$ROOT"
  echo "python=$PY"
  echo "data=$BASE/datasets/Haze4K/Haze4K"
} | tee -a "$STATUS"

cd "$ROOT"
git branch --show-current | tee -a "$STATUS"
git rev-parse HEAD | tee -a "$STATUS"
git status --short | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v3d_rarm_stage0_preflight.py \
  --checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$OUT" \
  --source_split train \
  --split_key val_inner \
  --max_samples 8 \
  > "$LOG" 2>&1
rc=$?
set -e

echo "stage0_preflight_done rc=$rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$rc" -eq 0 ]]; then
  echo "V3D_RARM_STAGE0_PREFLIGHT_OK" | tee -a "$STATUS"
else
  echo "V3D_RARM_STAGE0_PREFLIGHT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
