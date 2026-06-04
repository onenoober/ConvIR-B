#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf
ITS=$WORK/Dehazing/ITS
PY=/root/miniconda3/envs/convir-cu128/bin/python
DATA=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
A0=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
DEPTH=/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf
CKPT_DIR=$ITS/results/ConvIR-Haze4K-DPGA-Lite-v1.0-adapter-only-stop20-seed3407-20260604/Training-Results
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_lite_20260604
OUT_DIR=$LOG_DIR/eval_a0_compare
STATUS=$LOG_DIR/eval_status.txt

mkdir -p "$OUT_DIR"
cd "$ITS"

{
  echo "eval_start $(date -Is)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
} | tee "$STATUS"

run_eval() {
  local label="$1"
  local ckpt="$2"
  local tag="seed3407_${label}_vs_A0"
  local log="$LOG_DIR/eval_${tag}.log"

  echo "eval_${label}_start $(date -Is)" | tee -a "$STATUS"
  "$PY" "$WORK/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --original_arch convir \
    --original_mode original \
    --original_name a0 \
    --candidate_checkpoint "$ckpt" \
    --candidate_arch dpga \
    --candidate_mode original \
    --candidate_name "$label" \
    --dpga_depth_cache_dir "$DEPTH" \
    --dpga_eval_depth_split test \
    --output_dir "$OUT_DIR" \
    --tag "$tag" \
    2>&1 | tee "$log"
  echo "eval_${label}_done $(date -Is)" | tee -a "$STATUS"
}

run_eval dpga_lite_stop5 "$CKPT_DIR/model_5.pkl"
run_eval dpga_lite_stop20 "$CKPT_DIR/model_20.pkl"
run_eval dpga_lite_best "$CKPT_DIR/Best.pkl"
run_eval dpga_lite_final "$CKPT_DIR/Final.pkl"

echo "eval_done $(date -Is)" | tee -a "$STATUS"
