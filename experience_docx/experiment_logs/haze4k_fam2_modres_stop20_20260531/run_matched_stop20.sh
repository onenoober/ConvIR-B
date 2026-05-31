#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS

PY=/root/miniconda3/envs/convir-cu128/bin/python
DATA=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
ROOT=results/ConvIR-Haze4K-fam2-modres-stop20-20260531
STATUS="$ROOT/logs/status.txt"

mkdir -p "$ROOT/logs"
echo "start $(date -Iseconds)" > "$STATUS"

run_one() {
  local mode="$1"
  local name="ConvIR-Haze4K-${mode}-stop20-seed3407-20260531"
  local log="$ROOT/logs/${mode}_train_stop20_seed3407.log"
  local extra=()

  if [[ "$mode" == "fam2_modres" ]]; then
    extra+=(--mod_stats_freq 1 --mod_stats_batches 64)
  fi

  echo "running ${mode} $(date -Iseconds)" >> "$STATUS"
  "$PY" main.py \
    --model_name "$name" \
    --mode train \
    --version base \
    --fam_mode "$mode" \
    --seed 3407 \
    --data Haze4K \
    --data_dir "$DATA" \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    "${extra[@]}" > "$log" 2>&1
  echo "done ${mode} rc=$? $(date -Iseconds)" >> "$STATUS"
}

run_one original
run_one fam2_modres
echo "complete $(date -Iseconds)" >> "$STATUS"
