#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ITS="$ROOT/Dehazing/ITS"
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$OUT/status.txt"
MODEL_NAME="ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e1-seed3407-20260710"
MODEL_DIR="$ITS/results/$MODEL_NAME"
TRAIN_LOG="$OUT/v3d_stage1_1epoch_train.log"
AUDIT_LOG="$OUT/v3d_stage1_1epoch_audit.log"
FINAL_CKPT="$MODEL_DIR/Training-Results/Final.pkl"

mkdir -p "$OUT"
{
  echo "stage1_1epoch_start haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)"
  echo "root=$ROOT"
  echo "python=$PY"
  echo "gpu_id=$GPU_ID"
  echo "model_name=$MODEL_NAME"
} | tee -a "$STATUS"

if [ ! -e "$OUT/v3d_stage0_preflight_summary.json" ]; then
  echo "STAGE0_SUMMARY_MISSING_BLOCK" | tee -a "$STATUS"
  exit 30
fi
if ! "$PY" - "$OUT/v3d_stage0_preflight_summary.json" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
if not obj.get("pass"):
    raise SystemExit(1)
PY
then
  echo "STAGE0_NOT_PASS_BLOCK" | tee -a "$STATUS"
  exit 31
fi
if [ -e "$MODEL_DIR" ]; then
  echo "MODEL_DIR_EXISTS_BLOCK $MODEL_DIR" | tee -a "$STATUS"
  exit 32
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --arch convir \
  --fam_mode fam2_d7c_noop \
  --rarm_train_scope fam2_modulator_only \
  --allow_fam2_partial_init \
  --d7c_gate_mode d7c_fixed \
  --d7c_base_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --d7c_density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_need_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --mode train \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --weight_decay 0.0001 \
  --grad_clip_norm 0.001 \
  --num_epoch 1000 \
  --stop_epoch 1 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 1 \
  --valid_freq 999 \
  --init_model "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
echo "stage1_train_done rc=$train_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$train_rc" -ne 0 ]; then
  echo "V3D_RARM_STAGE1_TRAIN_FAILED" | tee -a "$STATUS"
  exit "$train_rc"
fi
if [ ! -e "$FINAL_CKPT" ]; then
  echo "FINAL_CKPT_MISSING_BLOCK $FINAL_CKPT" | tee -a "$STATUS"
  exit 33
fi

cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_chd_rm_v3d_stage1_smoke.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --candidate_checkpoint "$FINAL_CKPT" \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$OUT" \
  --source_split train \
  --split_key val_inner \
  --max_samples 64 \
  > "$AUDIT_LOG" 2>&1
audit_rc=$?
set -e
echo "stage1_audit_done rc=$audit_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$audit_rc" -eq 0 ]; then
  echo "V3D_RARM_STAGE1_1EPOCH_OK" | tee -a "$STATUS"
else
  echo "V3D_RARM_STAGE1_1EPOCH_FAILED" | tee -a "$STATUS"
fi
exit "$audit_rc"
