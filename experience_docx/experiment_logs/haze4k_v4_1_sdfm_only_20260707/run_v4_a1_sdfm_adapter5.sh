#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-1-sdfm-only
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_1_sdfm_only_20260707
SCRIPT=$EVID/run_v4_a1_sdfm_adapter5.sh
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
SESSION=v4a1_sdfm_train
MODEL_NAME=ConvIR-Haze4K-v4A1-SDFM-adapter-seed3407-20260707
TRAIN_LOG=$EVID/train_${MODEL_NAME}.log
OUT_DIR=$ITS/results/$MODEL_NAME
GPU_ID=${CUDA_VISIBLE_DEVICES:-2}

if [[ "${1:-}" == "--inside" ]]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  mkdir -p "$EVID"
  {
    echo "train_start v4_a1_sdfm_adapter5 $(date --iso-8601=seconds)"
    echo "work=$WORK"
    echo "model_name=$MODEL_NAME"
    echo "out_dir=$OUT_DIR"
    echo "data=$DATA"
    echo "a0=$A0"
    echo "python=$PY"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  } | tee -a "$STATUS"
  cd "$ITS"
  set +e
  PYTHONUNBUFFERED=1 "$PY" main.py \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --version base \
    --fam_mode original \
    --arch sfad_sdfm \
    --sfad_train_scope adapter_only \
    --mode train \
    --data_dir "$DATA" \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --grad_clip_norm 0.001 \
    --num_epoch 1000 \
    --stop_epoch 5 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --mod_stats_freq 1 \
    --mod_stats_batches 64 \
    --init_model "$A0" \
    --seed 3407 \
    > "$TRAIN_LOG" 2>&1
  rc=$?
  set -e
  echo "train_done rc=$rc v4_a1_sdfm_adapter5 $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$rc" -eq 0 ]]; then
    echo "V4_A1_SDFM_ADAPTER5_TRAIN_OK" | tee -a "$STATUS"
  else
    echo "V4_A1_SDFM_ADAPTER5_TRAIN_FAILED" | tee -a "$STATUS"
  fi
  exit "$rc"
fi

mkdir -p "$EVID"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "SESSION_ACTIVE $SESSION" | tee -a "$STATUS"
  exit 3
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "OUTPUT_EXISTS $OUT_DIR" | tee -a "$STATUS"
  exit 4
fi
echo "train_launch v4_a1_sdfm_adapter5 $(date --iso-8601=seconds) session=$SESSION gpu=$GPU_ID" | tee -a "$STATUS"
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU_ID bash '$SCRIPT' --inside"
echo "V4_A1_SDFM_ADAPTER5_LAUNCHED session=$SESSION" | tee -a "$STATUS"
