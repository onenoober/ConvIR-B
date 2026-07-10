#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3i-fam2-open-value-distillability}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-1}
ROUTE_ID=haze4k_v5_chd_rm_v3i_fam2_open_value_distillability_20260711
EVID="$REMOTE_ROOT/experience_docx/experiment_logs/$ROUTE_ID"
STATUS="$EVID/status.txt"
LOG="$EVID/v3i_d_disagreement_probe.log"

mkdir -p "$EVID"
echo "v3i_d_start $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$REMOTE_ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/chd_rm_v3i_d_disagreement_probe.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710/Training-Results/Final.pkl" \
  --peer_control_checkpoint "$BASE/repos/ConvIR-B-v3d-rarm-adapter-only-preflight/Dehazing/ITS/results/ConvIR-Haze4K-v3d-fam2modres-control-e1-seed3407-20260710/Training-Results/Final.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --v3i_a_policy_per_image "$EVID/v3i_a_policy_per_image.csv" \
  --output_dir "$EVID" \
  --max_samples 600 \
  --sample_per_image 192 \
  --fold_count 5 \
  --linear_steps 320 \
  --dw_steps 420 \
  --proj_channels 24 \
  --progress_every 25 \
  --log_every 100 \
  --v3i_b_best_mean_line 0.01662677374336075 \
  > "$LOG" 2>&1
rc=$?
set -e
echo "v3i_d_done rc=$rc $ROUTE_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo "V3I_D_DISAGREEMENT_PROBE_OK" | tee -a "$STATUS"
else
  echo "V3I_D_DISAGREEMENT_PROBE_FAILED" | tee -a "$STATUS"
fi
exit "$rc"
