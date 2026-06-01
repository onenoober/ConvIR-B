#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/workspace/ConvIR-B
ITS_ROOT="$ROOT/Dehazing/ITS"
DATA_DIR=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PY=/root/miniconda3/envs/convir-cu128/bin/python
RUN_ROOT="$ITS_ROOT/results/ConvIR-Haze4K-original-stop20-noise-floor-20260601"
LOG_DIR="$RUN_ROOT/logs"
STATUS="$LOG_DIR/status.txt"

read -r -a SEED_LIST <<< "${SEEDS:-3407 2027 8675}"

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

checkpoint_for_seed() {
  local seed="$1"
  if [[ "$seed" == "3407" ]]; then
    local existing="$ITS_ROOT/results/ConvIR-Haze4K-original-stop20-seed3407-20260531/Training-Results/Best.pkl"
    if [[ -f "$existing" ]]; then
      echo "$existing"
      return
    fi
  fi
  echo "$ITS_ROOT/results/ConvIR-Haze4K-original-stop20-seed${seed}-noise-floor-20260601/Training-Results/Best.pkl"
}

completion_marker_for_seed() {
  local seed="$1"
  if [[ "$seed" == "3407" ]]; then
    checkpoint_for_seed "$seed"
    return
  fi
  echo "$ITS_ROOT/results/ConvIR-Haze4K-original-stop20-seed${seed}-noise-floor-20260601/Training-Results/Final.pkl"
}

train_seed() {
  local seed="$1"
  local model_name="ConvIR-Haze4K-original-stop20-seed${seed}-noise-floor-20260601"
  local model_dir="$ITS_ROOT/results/$model_name"
  local checkpoint
  checkpoint="$(checkpoint_for_seed "$seed")"
  local completion_marker
  completion_marker="$(completion_marker_for_seed "$seed")"
  local log="$LOG_DIR/original_train_stop20_seed${seed}.log"

  if [[ -f "$completion_marker" ]]; then
    echo "reuse complete original seed ${seed} checkpoint $checkpoint $(date --iso-8601=seconds)" | tee -a "$STATUS"
    return
  fi
  if [[ -d "$model_dir" ]]; then
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    local interrupted_dir="${model_dir}-interrupted-${stamp}"
    echo "archive incomplete original seed ${seed} run to ${interrupted_dir} $(date --iso-8601=seconds)" | tee -a "$STATUS"
    mv "$model_dir" "$interrupted_dir"
    if [[ -f "$log" ]]; then
      mv "$log" "${log%.log}.interrupted-${stamp}.log"
    fi
  fi

  echo "running original seed ${seed} $(date --iso-8601=seconds)" | tee -a "$STATUS"
  "$PY" main.py \
    --mode train \
    --model_name "$model_name" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --fam_mode original \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed "$seed" \
    > "$log" 2>&1
  echo "done original seed ${seed} rc=$? $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

{
  echo "start $(date --iso-8601=seconds)"
  echo "seeds ${SEED_LIST[*]}"
} | tee "$STATUS"

for seed in "${SEED_LIST[@]}"; do
  train_seed "$seed"
done

eval_args=()
for seed in "${SEED_LIST[@]}"; do
  checkpoint="$(checkpoint_for_seed "$seed")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint for seed ${seed}: $checkpoint" | tee -a "$STATUS"
    exit 1
  fi
  eval_args+=(--run "${seed}:${checkpoint}")
done

echo "running original seed-noise eval $(date --iso-8601=seconds)" | tee -a "$STATUS"
"$PY" "$ROOT/experience_docx/tools/eval_haze4k_seed_noise.py" \
  --data_dir "$DATA_DIR" \
  --mode original \
  "${eval_args[@]}" \
  --reference_seed 3407 \
  --num_workers 0 \
  --output_json "$LOG_DIR/original_seed_noise_stop20.json" \
  --output_csv "$LOG_DIR/original_seed_noise_per_image.csv"

echo "complete $(date --iso-8601=seconds)" | tee -a "$STATUS"
