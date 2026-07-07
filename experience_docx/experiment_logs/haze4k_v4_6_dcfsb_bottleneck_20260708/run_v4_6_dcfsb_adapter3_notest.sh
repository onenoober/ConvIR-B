#!/usr/bin/env bash
set -euo pipefail
BASE=/sda/home/wangyuxin/ConvIR-B
WORK=$BASE/repos/ConvIR-B-haze4k-v4-6-dcfsb-bottleneck-independent
ITS=$WORK/Dehazing/ITS
EVID=$WORK/experience_docx/experiment_logs/haze4k_v4_6_dcfsb_bottleneck_20260708
SCRIPT=$EVID/run_v4_6_dcfsb_adapter3_notest.sh
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
TRAIN_SPLIT=$WORK/docs/ai_text_packages/haze4k_v4_sfad/splits/haze4k_train_adapter_train.txt
STATUS=$EVID/status.txt
SESSION=v46_dcfsb_train_adapter3
MODEL_NAME=ConvIR-Haze4K-v4A6-DCFSB-Bottleneck-adapter3-notest-seed3407-20260708
TRAIN_LOG=$EVID/train_${MODEL_NAME}.log
OUT_DIR=$ITS/results/$MODEL_NAME
GPU_ID=${CUDA_VISIBLE_DEVICES:-2}

if [[ "${1:-}" == "--inside" ]]; then
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  mkdir -p "$EVID"
  {
    echo "train_start v4_6_dcfsb_adapter3_notest $(date --iso-8601=seconds)"
    echo "work=$WORK"
    echo "model_name=$MODEL_NAME"
    echo "out_dir=$OUT_DIR"
    echo "data=$DATA"
    echo "train_split=$TRAIN_SPLIT"
    echo "a0=$A0"
    echo "python=$PY"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "locked_test_policy=no default validation; valid_freq=999 > stop_epoch=3"
  } | tee -a "$STATUS"
  cd "$ITS"
  set +e
  PYTHONUNBUFFERED=1 "$PY" main.py \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --version base \
    --fam_mode original \
    --arch dcfsb_bottleneck \
    --dcfsb_train_scope adapter_only \
    --mode train \
    --data_dir "$DATA" \
    --train_split_file "$TRAIN_SPLIT" \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --grad_clip_norm 0.001 \
    --num_epoch 1000 \
    --stop_epoch 3 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 3 \
    --valid_freq 999 \
    --mod_stats_freq 0 \
    --mod_stats_batches 0 \
    --init_model "$A0" \
    --seed 3407 \
    > "$TRAIN_LOG" 2>&1
  rc=$?
  set -e
  echo "train_done rc=$rc v4_6_dcfsb_adapter3_notest $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$rc" -eq 0 ]]; then
    echo "V4_6_DCFSB_ADAPTER3_NOTEST_TRAIN_OK" | tee -a "$STATUS"
  else
    echo "V4_6_DCFSB_ADAPTER3_NOTEST_TRAIN_FAILED" | tee -a "$STATUS"
  fi
  exit "$rc"
fi

mkdir -p "$EVID"
if ! grep -q 'V4_6_DCFSB_PREFLIGHT_OK' "$STATUS" 2>/dev/null; then
  echo "PREFLIGHT_NOT_OK v4_6_dcfsb" | tee -a "$STATUS"
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "SESSION_ACTIVE $SESSION" | tee -a "$STATUS"
  exit 3
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "OUTPUT_EXISTS $OUT_DIR" | tee -a "$STATUS"
  exit 4
fi
echo "train_launch v4_6_dcfsb_adapter3_notest $(date --iso-8601=seconds) session=$SESSION gpu=$GPU_ID" | tee -a "$STATUS"
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU_ID bash '$SCRIPT' --inside"
echo "V4_6_DCFSB_ADAPTER3_NOTEST_LAUNCHED session=$SESSION" | tee -a "$STATUS"
