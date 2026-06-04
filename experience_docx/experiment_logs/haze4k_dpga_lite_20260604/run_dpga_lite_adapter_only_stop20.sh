#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/convir-cu128/bin/python
DP=/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf
ITS=$DP/Dehazing/ITS
DATA=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
DEPTH=/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf
A0=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
LOG_DIR=$DP/experience_docx/experiment_logs/haze4k_dpga_lite_20260604
MODEL=ConvIR-Haze4K-DPGA-Lite-v1.0-adapter-only-stop20-seed3407-20260604
LOG=$LOG_DIR/train_${MODEL}.log
STATUS=$LOG_DIR/status.txt
{
  echo "train_start $MODEL $(date --iso-8601=seconds)"
  echo "root $DP"
  echo "data $DATA"
  echo "depth $DEPTH"
  echo "a0 $A0"
} | tee -a "$STATUS"
cd "$ITS"
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch dpga \
  --dpga_depth_cache_dir "$DEPTH" \
  --dpga_train_depth_split train \
  --dpga_eval_depth_split test \
  --dpga_train_scope adapter_only \
  --mode train \
  --data_dir "$DATA" \
  --batch_size 8 \
  --learning_rate 0.0004 \
  --weight_decay 0 \
  --num_epoch 1000 \
  --stop_epoch 20 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --mod_stats_freq 0 \
  --mod_stats_batches 64 \
  --init_model "$A0" \
  --seed 3407 \
  > "$LOG" 2>&1
rc=$?
echo "train_done $MODEL rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit $rc
