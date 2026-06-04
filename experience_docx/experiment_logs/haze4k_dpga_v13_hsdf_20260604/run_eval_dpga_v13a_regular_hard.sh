#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
SPLIT_JSON=${SPLIT_JSON:-$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
MODEL_NAME=ConvIR-Haze4K-DPGA-v1.3A-HSDF-lossmask-hardaware-shallow-scale0p25-hardw1p2-t0p01-seed3407-20260604
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
OUT=$LOG_DIR/v13a_eval_regular_hard
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT"
cd "$WORK/Dehazing/ITS"

for ckpt in Best Final; do
  for split in val_regular val_hard; do
    PYTHONUNBUFFERED=1 "$PY" ../../experience_docx/tools/eval_haze4k_checkpoint_compare.py \
      --data_dir "$DATA" \
      --original_checkpoint "$A0" \
      --original_arch convir \
      --original_mode original \
      --original_name a0 \
      --candidate_checkpoint "$MODEL_DIR/$ckpt.pkl" \
      --candidate_arch dpga \
      --candidate_mode "$ckpt" \
      --candidate_name "v13a_${ckpt,,}_${split}" \
      --candidate_dpga_active_adapters shallow \
      --candidate_dpga_scale_multiplier 0.25 \
      --dpga_depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --split_name "$split" \
      --output_dir "$OUT" \
      --tag "v13a_${ckpt,,}_${split}_vs_a0" \
      > "$OUT/eval_v13a_${ckpt,,}_${split}.log" 2>&1
  done
done

"$PY" ../../experience_docx/tools/gate_haze4k_dpga_v13_regular_hard.py \
  --best_regular_compare_json "$OUT/scout_eval_compare_v13a_best_val_regular_vs_a0.json" \
  --best_hard_compare_json "$OUT/scout_eval_compare_v13a_best_val_hard_vs_a0.json" \
  --final_regular_compare_json "$OUT/scout_eval_compare_v13a_final_val_regular_vs_a0.json" \
  --final_hard_compare_json "$OUT/scout_eval_compare_v13a_final_val_hard_vs_a0.json" \
  --output "$OUT/dpga_v13_gate_eval_regular_and_hard.json" \
  > "$OUT/gate_v13a_regular_hard.log" 2>&1 || true

echo "v13a_eval_regular_hard_done output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
