#!/usr/bin/env bash
set -euo pipefail

BASE=/sda/home/wangyuxin/ConvIR-B
ROOT=$BASE/repos/ConvIR-B-haze4k-v5-v3a-d7c-gated-noop-connection-audit
EVID=$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3a_d7c_gated_noop_connection_audit_20260710
PY=$BASE/envs/convir-cu121/bin/python
STATUS=$EVID/status.txt
LOG=$EVID/v3a_d7c_gated_noop_connection.log

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$EVID"
echo "v3a_start $(date --iso-8601=seconds) cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$STATUS"

cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/run_chd_rm_v3a_d7c_gated_noop_connection.py \
  --checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$EVID" \
  --source_split train \
  --real_batch_split val_inner \
  --full_split val_inner \
  --real_batch_size 4 \
  --progress_every 100 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "v3a_done rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$rc" -eq 0 ]; then
  echo V3A_D7C_GATED_NOOP_CONNECTION_OK | tee -a "$STATUS"
else
  echo V3A_D7C_GATED_NOOP_CONNECTION_FAILED | tee -a "$STATUS"
fi
exit "$rc"
