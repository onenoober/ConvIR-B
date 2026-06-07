#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/workspace/ConvIR-B-v04e-repro-826caaf-bundle
ITS_ROOT=$ROOT/Dehazing/ITS
PY=/root/miniconda3/envs/convir-cu128/bin/python
DATA_DIR=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PRETRAIN=/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl
A1DIR=/root/autodl-tmp/workspace/ConvIR-B-pfd-mainline/Dehazing/ITS/results/ConvIR-Haze4K-A1-init-official-stop20-seed3407-20260602/Training-Results
LOG_DIR=$ROOT/experience_docx/experiment_logs/haze4k_rootcause_preexp_20260604
mkdir -p "$LOG_DIR"
{
  echo "a1_vs_a0_compare_start $(date --iso-8601=seconds)"
  echo "root $ROOT"
  echo "pretrain $PRETRAIN"
  echo "a1dir $A1DIR"
} | tee -a "$LOG_DIR/status.txt"
cd "$ITS_ROOT"
for item in a1_stop5:model_5.pkl a1_best:Best.pkl a1_final:Final.pkl; do
  tag=${item%%:*}
  ckpt_name=${item#*:}
  ckpt="$A1DIR/$ckpt_name"
  echo "fixed_compare_start $tag $ckpt $(date --iso-8601=seconds)" | tee -a "$LOG_DIR/status.txt"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$PRETRAIN" \
    --original_arch convir --original_mode original --original_name a0_official \
    --candidate_checkpoint "$ckpt" \
    --candidate_arch convir --candidate_mode original --candidate_name "$tag" \
    --output_dir "$LOG_DIR" \
    --tag "${tag}_vs_a0" \
    > "$LOG_DIR/${tag}_vs_a0_compare.log" 2>&1
  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_${tag}_vs_a0.csv" \
    --candidate_name "$tag" \
    --output "$LOG_DIR/scout_eval_bucket_analysis_${tag}_vs_a0.json" \
    > "$LOG_DIR/${tag}_vs_a0_buckets.log" 2>&1
  echo "fixed_compare_done $tag $(date --iso-8601=seconds)" | tee -a "$LOG_DIR/status.txt"
done
echo "a1_vs_a0_compare_complete $(date --iso-8601=seconds)" | tee -a "$LOG_DIR/status.txt"
