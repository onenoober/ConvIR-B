#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-4b-bidpfm1}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
MODEL_NAME=${MODEL_NAME:-ConvIR-Haze4K-v1.4B-BiDPFM1-adapter-only-seed3407-20260604}
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
OUT=$LOG_DIR/v14b_eval_regular_hard
STATUS=$LOG_DIR/status.txt

if tmux has-session -t v14b_bidpfm1_train 2>/dev/null; then
  echo "v14b eval blocked: training session v14b_bidpfm1_train is still active" >&2
  exit 5
fi
for ckpt in Best Final; do
  if [[ ! -f "$MODEL_DIR/$ckpt.pkl" ]]; then
    echo "v14b eval blocked: missing checkpoint $MODEL_DIR/$ckpt.pkl" >&2
    exit 3
  fi
done

mkdir -p "$OUT"
echo "v14b_eval_regular_hard_start model=$MODEL_NAME output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
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
      --candidate_name "v14b_${ckpt,,}_${split}" \
      --candidate_dpga_fusion_mode udp_bi \
      --candidate_dpga_active_adapters dpfm1 \
      --candidate_dpga_udp_components all \
      --candidate_dpga_scale_multiplier 1.0 \
      --dpga_depth_cache_dir "$DEPTH" \
      --split_json "$SPLIT_JSON" \
      --split_name "$split" \
      --output_dir "$OUT" \
      --tag "v14b_${ckpt,,}_${split}_vs_a0" \
      > "$OUT/eval_v14b_${ckpt,,}_${split}.log" 2>&1
  done
done

"$PY" ../../experience_docx/tools/gate_haze4k_v14b_bidpfm1.py \
  --best_regular_compare_json "$OUT/scout_eval_compare_v14b_best_val_regular_vs_a0.json" \
  --best_hard_compare_json "$OUT/scout_eval_compare_v14b_best_val_hard_vs_a0.json" \
  --final_regular_compare_json "$OUT/scout_eval_compare_v14b_final_val_regular_vs_a0.json" \
  --final_hard_compare_json "$OUT/scout_eval_compare_v14b_final_val_hard_vs_a0.json" \
  --output "$OUT/v14b_gate_eval_regular_and_hard.json" \
  > "$OUT/gate_v14b_regular_hard.log" 2>&1 || true

echo "v14b_eval_regular_hard_done output=$OUT $(date --iso-8601=seconds)" | tee -a "$STATUS"
