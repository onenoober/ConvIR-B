#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-4-udp-lite}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_udp_lite_v14_20260604
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
MODEL_NAME=${MODEL_NAME:-ConvIR-Haze4K-v1.4A-UDP-Lite-DPFM123-adapter-only-seed3407-20260604}
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
OUT=$LOG_DIR/v14a_intermediates
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT"
cd "$WORK"

for split in val_regular val_hard; do
  SPLIT_OUT="$OUT/$split"
  mkdir -p "$SPLIT_OUT"
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_udp_lite_intermediates.py \
    --its_dir Dehazing/ITS \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --candidate_checkpoint "best=$MODEL_DIR/Best.pkl" \
    --candidate_checkpoint "final=$MODEL_DIR/Final.pkl" \
    --dpga_depth_cache_dir "$DEPTH" \
    --dpga_eval_depth_split train \
    --split_json "$SPLIT_JSON" \
    --split_name "$split" \
    --output_dir "$SPLIT_OUT" \
    > "$SPLIT_OUT/v14_intermediate_audit_${split}.log" 2>&1
done

echo "v14_intermediate_audits_done output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
