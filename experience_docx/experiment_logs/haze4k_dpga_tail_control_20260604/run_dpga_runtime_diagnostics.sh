#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
LITE_WORK=${LITE_WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-lite-826caaf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
CKPT_DIR=${CKPT_DIR:-$LITE_WORK/Dehazing/ITS/results/ConvIR-Haze4K-DPGA-Lite-v1.0-adapter-only-stop20-seed3407-20260604/Training-Results}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
OUT_DIR=$LOG_DIR/runtime_diagnostics
LOG=$LOG_DIR/dpga_runtime_diagnostics.log
STATUS=$LOG_DIR/status.txt

mkdir -p "$OUT_DIR"
cd "$WORK/Dehazing/ITS"

{
  echo "dpga_runtime_diagnostics_start $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "lite_work=$LITE_WORK"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "ckpt_dir=$CKPT_DIR"
  "$PY" "$WORK/experience_docx/tools/audit_haze4k_dpga_runtime_variants.py" \
    --its_dir "$WORK/Dehazing/ITS" \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --candidate_checkpoint best="$CKPT_DIR/Best.pkl" \
    --candidate_checkpoint final="$CKPT_DIR/Final.pkl" \
    --dpga_depth_cache_dir "$DEPTH" \
    --dpga_eval_depth_split test \
    --output_dir "$OUT_DIR"
  echo "dpga_runtime_diagnostics_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"

{
  echo "runtime_diagnostics_out=$OUT_DIR"
  echo "module_ablation_csv=$OUT_DIR/dpga_module_ablation_best_final.csv"
  echo "scale_sweep_csv=$OUT_DIR/dpga_scale_sweep_best_final.csv"
} | tee -a "$STATUS"
