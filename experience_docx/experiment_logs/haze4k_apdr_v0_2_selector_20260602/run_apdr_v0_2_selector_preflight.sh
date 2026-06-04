#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-convir-v0-2}
ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2_selector_20260602"
STATUS="$LOG_DIR/status.txt"
TAG=${TAG:-apdr_v0_2_selector_seed3407}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
TRAIN_BATCHES_PER_EPOCH=${TRAIN_BATCHES_PER_EPOCH:-0}
CALIBRATION_IMAGES=${CALIBRATION_IMAGES:-0}
LOSS_EVAL_IMAGES=${LOSS_EVAL_IMAGES:-128}
PIXEL_SAMPLES_PER_IMAGE=${PIXEL_SAMPLES_PER_IMAGE:-2048}
PROGRESS_FREQ=${PROGRESS_FREQ:-100}

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start root=$ROOT tag=$TAG"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_apdr.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output "$LOG_DIR/preflight_apdr_v0_2_arch.json" \
  --stage apdr_v0_2_arch_preflight \
  --device cuda \
  --height 256 \
  --width 256 \
  --batch_size 1 \
  --apdr_selector_mode v0_2 \
  --apdr_active_scales full \
  --apdr_loss_scales full_only \
  --apdr_gate_init 0.01 \
  --apdr_anchor_lambda 0.10 \
  --apdr_gate_supervision_lambda 0.02 \
  --apdr_gate_lambda 0.002 \
  --apdr_residual_lambda 0.02 \
  --apdr_risk_temperature 5.0
log_status "arch_preflight_pass"

log_status "selector_preflight_start epochs=$EPOCHS train_batches=$TRAIN_BATCHES_PER_EPOCH"
"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_apdr_v0_2_selector.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output_dir "$LOG_DIR" \
  --tag "$TAG" \
  --device cuda \
  --seed 3407 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_worker 8 \
  --learning_rate 2e-4 \
  --train_batches_per_epoch "$TRAIN_BATCHES_PER_EPOCH" \
  --calibration_images "$CALIBRATION_IMAGES" \
  --loss_eval_images "$LOSS_EVAL_IMAGES" \
  --pixel_samples_per_image "$PIXEL_SAMPLES_PER_IMAGE" \
  --progress_freq "$PROGRESS_FREQ" \
  > "$LOG_DIR/selector_preflight_${TAG}.log" 2>&1

if "$PY" - "$LOG_DIR/gate_${TAG}.json" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(gate, indent=2), flush=True)
raise SystemExit(0 if gate.get("pass") else 1)
PY
then
  log_status "selector_gate_pass $TAG"
else
  log_status "selector_gate_fail_stop $TAG"
  exit 0
fi

log_status "complete_selector_only $TAG"
