#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3f-operator-correctability-ranker}
V3D=${V3D:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ROUTE_ID=haze4k_v5_chd_rm_v3f_operator_correctability_ranker_20260710
EVID="$ROOT/experience_docx/experiment_logs/$ROUTE_ID"
STATUS="$EVID/status.txt"
LOG="$EVID/v3f_a_correctability_audit.log"

mkdir -p "$EVID"
echo "v3f_a_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3f_correctability_audit.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --d7c_checkpoint "$V3D/Dehazing/ITS/results/ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --control_checkpoint "$V3D/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$EVID" \
  --max_samples 600 \
  --pixel_sample_per_image 8192 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3f_a_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3F_A_CORRECTABILITY_AUDIT_OK" | tee -a "$STATUS"
else
  echo "V3F_A_CORRECTABILITY_AUDIT_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
