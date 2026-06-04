#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
SPLIT_JSON=${SPLIT_JSON:-$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
MODEL_NAME=ConvIR-Haze4K-DPGA-v1.3B-HSDF-hardgated-bottleneck-shallow-scale0p25-bneckmax0p05-gatelambda0p01-seed3407-20260604
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
OUT=$LOG_DIR/runtime_diagnostics_v13b_val_regula
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT"
cd "$WORK"
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dpga_runtime_variants.py \
  --its_dir Dehazing/ITS \
  --data_dir "$DATA" \
  --original_checkpoint "$A0" \
  --candidate_checkpoint "best=$MODEL_DIR/Best.pkl" \
  --candidate_checkpoint "final=$MODEL_DIR/Final.pkl" \
  --dpga_depth_cache_dir "$DEPTH" \
  --dpga_eval_depth_split train \
  --split_json "$SPLIT_JSON" \
  --split_name val_regular \
  --dpga_hard_gate_mode bottleneck \
  --dpga_bottleneck_scale_multiplier 2.0 \
  --dpga_module_scale_multiplier 0.25 \
  --output_dir "$OUT" \
  --run_module_ablation \
  > "$LOG_DIR/dpga_v13b_runtime_ablation_val_regular.log" 2>&1

cp "$OUT/dpga_module_ablation_best_final.csv" "$OUT/dpga_v13b_runtime_ablation_on_val_inner.csv"
echo "v13b_runtime_ablation_done output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
