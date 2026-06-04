#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-dpga-tail-control}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR=$WORK/experience_docx/experiment_logs/haze4k_dpga_tail_control_20260604
SPLIT_JSON=$LOG_DIR/internal_val/haze4k_train_inner_val_inner_seed3407.json
DECISION_JSON=$LOG_DIR/v1_1_decision/dpga_v1_1_training_decision.json
STATUS=$LOG_DIR/status.txt

if [[ ! -f "$DECISION_JSON" ]]; then
  echo "missing decision json: $DECISION_JSON" >&2
  exit 2
fi
if [[ ! -f "$SPLIT_JSON" ]]; then
  echo "missing internal val split: $SPLIT_JSON" >&2
  exit 2
fi

eval "$("$PY" - "$DECISION_JSON" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
decision = json.load(open(path, "r", encoding="utf-8"))
if not decision.get("launch_allowed"):
    print("echo 'decision does not allow launch' >&2")
    print("exit 3")
    raise SystemExit(0)
args = decision["training_args"]
keys = [
    "model_name",
    "dpga_active_adapters",
    "dpga_scale_multiplier",
    "dpga_adapter_residual_scale",
    "dpga_tc_rec_loss",
    "dpga_tc_fft_lambda",
    "dpga_tc_anchor_lambda",
    "dpga_tc_chroma_lambda",
    "dpga_tc_delta_lambda",
    "dpga_tc_delta_tv_lambda",
    "dpga_tc_anchor_error_threshold",
    "learning_rate",
    "weight_decay",
    "stop_epoch",
    "seed",
]
for key in keys:
    value = args[key]
    print(f"{key.upper()}={shlex.quote(str(value))}")
PY
)"

ITS=$WORK/Dehazing/ITS
MODEL_DIR=$ITS/results/$MODEL_NAME
TRAIN_LOG=$LOG_DIR/train_${MODEL_NAME}.log

if [[ -e "$MODEL_DIR/Training-Results/Final.pkl" ]]; then
  echo "training final checkpoint already exists: $MODEL_DIR/Training-Results/Final.pkl" >&2
  exit 4
fi

{
  echo "v1_1_train_start $MODEL_NAME $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "split_json=$SPLIT_JSON"
  echo "decision_json=$DECISION_JSON"
  echo "active_adapters=$DPGA_ACTIVE_ADAPTERS"
  echo "scale_multiplier=$DPGA_SCALE_MULTIPLIER"
  echo "tail_control=rec:$DPGA_TC_REC_LOSS fft:$DPGA_TC_FFT_LAMBDA anchor:$DPGA_TC_ANCHOR_LAMBDA chroma:$DPGA_TC_CHROMA_LAMBDA delta:$DPGA_TC_DELTA_LAMBDA tv:$DPGA_TC_DELTA_TV_LAMBDA"
} | tee -a "$STATUS"

cd "$ITS"
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
  --dpga_valid_split_name val_inner \
  --dpga_train_scope adapter_only \
  --dpga_active_adapters "$DPGA_ACTIVE_ADAPTERS" \
  --dpga_scale_multiplier "$DPGA_SCALE_MULTIPLIER" \
  --dpga_adapter_residual_scale "$DPGA_ADAPTER_RESIDUAL_SCALE" \
  --dpga_tc_rec_loss "$DPGA_TC_REC_LOSS" \
  --dpga_tc_fft_lambda "$DPGA_TC_FFT_LAMBDA" \
  --dpga_tc_anchor_lambda "$DPGA_TC_ANCHOR_LAMBDA" \
  --dpga_tc_chroma_lambda "$DPGA_TC_CHROMA_LAMBDA" \
  --dpga_tc_delta_lambda "$DPGA_TC_DELTA_LAMBDA" \
  --dpga_tc_delta_tv_lambda "$DPGA_TC_DELTA_TV_LAMBDA" \
  --dpga_tc_anchor_error_threshold "$DPGA_TC_ANCHOR_ERROR_THRESHOLD" \
  --mode train \
  --data_dir "$DATA" \
  --batch_size 8 \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip_norm 0.001 \
  --num_epoch 1000 \
  --stop_epoch "$STOP_EPOCH" \
  --print_freq 50 \
  --num_worker 8 \
  --save_freq 5 \
  --valid_freq 1 \
  --mod_stats_freq 1 \
  --mod_stats_batches 64 \
  --init_model "$A0" \
  --seed "$SEED" \
  > "$TRAIN_LOG" 2>&1
rc=$?
echo "v1_1_train_done $MODEL_NAME rc=$rc $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
