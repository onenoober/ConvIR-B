#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ITS="$ROOT/Dehazing/ITS"
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$OUT/status.txt"
SRC_MODEL="ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e1-seed3407-20260710"
MODEL_NAME="ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e5frome1-seed3407-20260710"
MODEL_DIR="$ITS/results/$MODEL_NAME"
RESUME_CKPT="$ITS/results/$SRC_MODEL/Training-Results/model.pkl"
FINAL_CKPT="$MODEL_DIR/Training-Results/Final.pkl"
TRAIN_LOG="$OUT/v3d_stage1_5epoch_train.log"
AUDIT_LOG="$OUT/v3d_stage1_5epoch_audit.log"

mkdir -p "$OUT"
{
  echo "stage1_5epoch_start haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)"
  echo "root=$ROOT"
  echo "python=$PY"
  echo "gpu_id=$GPU_ID"
  echo "model_name=$MODEL_NAME"
  echo "resume_checkpoint=$RESUME_CKPT"
} | tee -a "$STATUS"

if [ ! -e "$OUT/v3d_stage1_5epoch_decision.json" ]; then
  echo "STAGE1_5EPOCH_DECISION_MISSING_BLOCK" | tee -a "$STATUS"
  exit 40
fi
if ! "$PY" - "$OUT/v3d_stage1_5epoch_decision.json" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
if not obj.get("authorized") or obj.get("locked_test_authorized"):
    raise SystemExit(1)
PY
then
  echo "STAGE1_5EPOCH_DECISION_NOT_AUTHORIZED_BLOCK" | tee -a "$STATUS"
  exit 41
fi
if [ ! -e "$RESUME_CKPT" ]; then
  echo "RESUME_CKPT_MISSING_BLOCK $RESUME_CKPT" | tee -a "$STATUS"
  exit 42
fi
if [ -e "$MODEL_DIR" ]; then
  echo "MODEL_DIR_EXISTS_BLOCK $MODEL_DIR" | tee -a "$STATUS"
  exit 43
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
  --stop_epoch 5 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 999 \
  --resume "$RESUME_CKPT" \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
echo "stage1_5epoch_train_done rc=$train_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$train_rc" -ne 0 ]; then
  echo "V3D_RARM_STAGE1_5EPOCH_TRAIN_FAILED" | tee -a "$STATUS"
  exit "$train_rc"
fi
if [ ! -e "$FINAL_CKPT" ]; then
  echo "FINAL_CKPT_MISSING_BLOCK $FINAL_CKPT" | tee -a "$STATUS"
  exit 44
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
  --max_samples 600 \
  --run_label stage1_5epoch \
  --decision_pass V3D_RARM_STAGE1_5EPOCH_PASS_REQUIRE_POST5_DECISION_NO_DIRECT_V4 \
  --decision_fail V3D_RARM_STAGE1_5EPOCH_FAIL_NO_CONTINUATION \
  --next_action_pass "Write a separate post-5epoch decision before any 20-epoch, neighbor, v4, or locked-test step." \
  --next_action_fail "Stop v3d training continuation and inspect Stage 1 5-epoch failure." \
  > "$AUDIT_LOG" 2>&1
audit_rc=$?
set -e
echo "stage1_5epoch_audit_done rc=$audit_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$audit_rc" -eq 0 ]; then
  echo "V3D_RARM_STAGE1_5EPOCH_OK" | tee -a "$STATUS"
else
  echo "V3D_RARM_STAGE1_5EPOCH_FAILED" | tee -a "$STATUS"
fi
exit "$audit_rc"
