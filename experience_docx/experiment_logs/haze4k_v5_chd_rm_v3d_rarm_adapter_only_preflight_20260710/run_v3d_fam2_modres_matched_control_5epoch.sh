#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3d-rarm-adapter-only-preflight}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
PY=${PY:-/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python}
GPU_ID=${GPU_ID:-0}
ITS="$ROOT/Dehazing/ITS"
OUT="$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710"
STATUS="$OUT/status.txt"
DECISION="$OUT/v3d_post5_matched_control_decision.json"
MODEL_E1="ConvIR-Haze4K-v3d-fam2modres-control-e1-seed3407-20260710"
MODEL_E5="ConvIR-Haze4K-v3d-fam2modres-control-e5frome1-seed3407-20260710"
MODEL_E1_DIR="$ITS/results/$MODEL_E1"
MODEL_E5_DIR="$ITS/results/$MODEL_E5"
RESUME_CKPT="$MODEL_E1_DIR/Training-Results/model.pkl"
FINAL_CKPT="$MODEL_E5_DIR/Training-Results/Final.pkl"
TRAIN_E1_LOG="$OUT/v3d_fam2modres_control_1epoch_train.log"
TRAIN_E5_LOG="$OUT/v3d_fam2modres_control_5epoch_train.log"
AUDIT_LOG="$OUT/v3d_fam2modres_control_5epoch_audit.log"

mkdir -p "$OUT"
{
  echo "fam2modres_control_start haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)"
  echo "root=$ROOT"
  echo "python=$PY"
  echo "gpu_id=$GPU_ID"
  echo "model_e1=$MODEL_E1"
  echo "model_e5=$MODEL_E5"
} | tee -a "$STATUS"

if [ ! -e "$DECISION" ]; then
  echo "MATCHED_CONTROL_DECISION_MISSING_BLOCK" | tee -a "$STATUS"
  exit 50
fi
if ! "$PY" - "$DECISION" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
if not obj.get("authorized") or obj.get("locked_test_authorized"):
    raise SystemExit(1)
PY
then
  echo "MATCHED_CONTROL_NOT_AUTHORIZED_BLOCK" | tee -a "$STATUS"
  exit 51
fi
if [ -e "$MODEL_E1_DIR" ] || [ -e "$MODEL_E5_DIR" ]; then
  echo "CONTROL_MODEL_DIR_EXISTS_BLOCK" | tee -a "$STATUS"
  exit 52
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_E1" \
  --data Haze4K \
  --version base \
  --arch convir \
  --fam_mode fam2_modres \
  --rarm_train_scope fam2_modulator_only \
  --allow_fam2_partial_init \
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
  > "$TRAIN_E1_LOG" 2>&1
e1_rc=$?
set -e
echo "fam2modres_control_e1_done rc=$e1_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$e1_rc" -ne 0 ]; then
  echo "V3D_FAM2MODRES_CONTROL_E1_FAILED" | tee -a "$STATUS"
  exit "$e1_rc"
fi
if [ ! -e "$RESUME_CKPT" ]; then
  echo "CONTROL_RESUME_CKPT_MISSING_BLOCK $RESUME_CKPT" | tee -a "$STATUS"
  exit 53
fi

set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_E5" \
  --data Haze4K \
  --version base \
  --arch convir \
  --fam_mode fam2_modres \
  --rarm_train_scope fam2_modulator_only \
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
  > "$TRAIN_E5_LOG" 2>&1
e5_rc=$?
set -e
echo "fam2modres_control_e5_done rc=$e5_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$e5_rc" -ne 0 ]; then
  echo "V3D_FAM2MODRES_CONTROL_E5_FAILED" | tee -a "$STATUS"
  exit "$e5_rc"
fi
if [ ! -e "$FINAL_CKPT" ]; then
  echo "CONTROL_FINAL_CKPT_MISSING_BLOCK $FINAL_CKPT" | tee -a "$STATUS"
  exit 54
fi

cd "$ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_chd_rm_v3d_stage1_smoke.py \
  --a0_checkpoint "$BASE/checkpoints/official/Haze4K/haze4k-base.pkl" \
  --candidate_checkpoint "$FINAL_CKPT" \
  --candidate_fam_mode fam2_modres \
  --data_dir "$BASE/datasets/Haze4K/Haze4K" \
  --split_json "$ROOT/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json" \
  --density_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt" \
  --d7c_artifact "$BASE/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt" \
  --output_dir "$OUT" \
  --source_split train \
  --split_key val_inner \
  --max_samples 600 \
  --run_label fam2modres_control_5epoch \
  --decision_pass V3D_FAM2MODRES_CONTROL_5EPOCH_DONE_REQUIRE_D7C_COMPARISON \
  --decision_fail V3D_FAM2MODRES_CONTROL_5EPOCH_FAIL \
  --next_action_pass "Compare matched FAM2 modres control against D7c RARM before any longer run." \
  --next_action_fail "Stop and inspect FAM2 modres control failure." \
  > "$AUDIT_LOG" 2>&1
audit_rc=$?
set -e
echo "fam2modres_control_audit_done rc=$audit_rc haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [ "$audit_rc" -eq 0 ]; then
  echo "V3D_FAM2MODRES_CONTROL_5EPOCH_OK" | tee -a "$STATUS"
else
  echo "V3D_FAM2MODRES_CONTROL_5EPOCH_FAILED" | tee -a "$STATUS"
fi
exit "$audit_rc"
