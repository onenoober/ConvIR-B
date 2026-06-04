#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-3-hsdf}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604
SPLIT_JSON=${SPLIT_JSON:-$LOG_DIR/intenal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
STATUS=$LOG_DIR/status.txt

MODEL_NAME=ConvIR-Haze4K-DPGA-v1.3A-HSDF-lossmask-hardaware-shallow-scale0p25-hardw1p2-t0p01-seed3407-20260604
MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME
TRAIN_LOG=$LOG_DIR/train_${MODEL_NAME}.log

if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "missing split json: $SPLIT_JSON" >&2
  exit 2
fi
if [[ -e "$MODEL_DIR/Training-Results/Final.pkl" ]]; then
  echo "training final checkpoint already exists: $MODEL_DIR/Training-Results/Final.pkl" >&2
  exit 4
fi

{
  echo "v13a_train_start $MODEL_NAME $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "split_json=$SPLIT_JSON"
  echo "active_adapters=shallow scale=0.25 mask=hard_selective hard_sample_lambda=0.20 hard_region_lambda=0.35 hard_region_error_threshold=0.010"
} | tee -a "$STATUS"

cd "$WORK/Dehazing/ITS"
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch dpga \
  --dpga_depth_cache_dir "$DEPTH" \
  --dpga_train_depth_split train \
  --dpga_eval_depth_split train \
  --dpga_train_split_json "$SPLIT_JSON" \
  --dpga_train_split_name train_inner \
  --dpga_valid_split_json "$SPLIT_JSON" \
  --dpga_valid_split_name val_regular \
  --dpga_train_scope adapter_only \
  --dpga_active_adapters shallow \
  --dpga_scale_multiplier 0.25 \
  --dpga_adapter_residual_scale 0.1 \
  --dpga_tc_rec_loss charbonnier \
  --dpga_tc_fft_lambda 0.05 \
  --dpga_tc_anchor_lambda 0.08 \
  --dpga_tc_chroma_lambda 0.03 \
  --dpga_tc_delta_lambda 0.0002 \
  --dpga_tc_delta_tv_lambda 0.00005 \
  --dpga_tc_anchor_error_threshold 0.035 \
  --dpga_tc_mask_mode hard_selective \
  --dpga_hard_sample_lambda 0.20 \
  --dpga_hard_region_lambda 0.35 \
  --dpga_hard_region_error_threshold 0.010 \
  --dpga_require_hard_labels 1 \
  --dpga_hard_sampler_json "$SPLIT_JSON" \
  --dpga_hard_sampler_split_name train_inner \
  --dpga_hard_sampler_seed 3407 \
  --dpga_hard_sampler_hard_ratio 0.3333333333 \
  --dpga_hard_sampler_medium_ratio 0.3333333333 \
  --mode train \
  --data_dir "$DATA" \
  --batch_size 8 \
  --leaning_rate 0.0003 \
  --weight_decay 0.0001 \
  --grad_clip_norm 0.001 \
  --num_epoch 1000 \
  --stop_epoch 20 \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --mod_stats_freq 1 \
  --mod_stats_batches 64 \
  --init_model "$A0" \
  --seed 3407 \
  > "$TRAIN_LOG" 2>&1

echo "v13a_train_done rc=$? $MODEL_NAME $(date --iso-8601=seconds)" | tee -a "$STATUS"
