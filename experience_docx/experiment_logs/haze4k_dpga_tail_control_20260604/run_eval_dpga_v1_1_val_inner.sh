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
OUT_DIR=$LOG_DIR/v1_1_val_inner_eval
LOG=$LOG_DIR/eval_dpga_v1_1_val_inner.log
STATUS=$LOG_DIR/status.txt

if [[ ! -f "$DECISION_JSON" ]]; then
  echo "missing decision json: $DECISION_JSON" >&2
  exit 2
fi

eval "$("$PY" - "$DECISION_JSON" <<'PY'
import json
import shlex
import sys

decision = json.load(open(sys.argv[1], "r", encoding="utf-8"))
args = decision["training_args"]
for key in ("model_name", "dpga_active_adapters", "dpga_scale_multiplier", "dpga_adapter_residual_scale"):
    print(f"{key.upper()}={shlex.quote(str(args[key]))}")
PY
)"

CKPT_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
BEST=$CKPT_DIR/Best.pkl
FINAL=$CKPT_DIR/Final.pkl
if [[ ! -f "$BEST" || ! -f "$FINAL" ]]; then
  echo "missing v1.1 checkpoints: best=$BEST final=$FINAL" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"

{
  echo "eval_dpga_v1_1_val_inner_start $(date --iso-8601=seconds)"
  echo "model_name=$MODEL_NAME"
  echo "best=$BEST"
  echo "final=$FINAL"
  echo "split_json=$SPLIT_JSON"
  cd "$WORK/Dehazing/ITS"
  "$PY" "$WORK/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --original_arch convir \
    --original_mode original \
    --original_name A0 \
    --candidate_checkpoint "$BEST" \
    --candidate_arch dpga \
    --candidate_mode original \
    --candidate_name dpga_v1_1_best \
    --dpga_depth_cache_dir "$DEPTH" \
    --dpga_eval_depth_split train \
    --split_json "$SPLIT_JSON" \
    --split_name val_inner \
    --candidate_dpga_active_adapters "$DPGA_ACTIVE_ADAPTERS" \
    --candidate_dpga_scale_multiplier "$DPGA_SCALE_MULTIPLIER" \
    --candidate_dpga_adapter_residual_scale "$DPGA_ADAPTER_RESIDUAL_SCALE" \
    --output_dir "$OUT_DIR" \
    --tag v1_1_val_inner_best_vs_a0
  "$PY" "$WORK/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --original_arch convir \
    --original_mode original \
    --original_name A0 \
    --candidate_checkpoint "$FINAL" \
    --candidate_arch dpga \
    --candidate_mode original \
    --candidate_name dpga_v1_1_final \
    --dpga_depth_cache_dir "$DEPTH" \
    --dpga_eval_depth_split train \
    --split_json "$SPLIT_JSON" \
    --split_name val_inner \
    --candidate_dpga_active_adapters "$DPGA_ACTIVE_ADAPTERS" \
    --candidate_dpga_scale_multiplier "$DPGA_SCALE_MULTIPLIER" \
    --candidate_dpga_adapter_residual_scale "$DPGA_ADAPTER_RESIDUAL_SCALE" \
    --output_dir "$OUT_DIR" \
    --tag v1_1_val_inner_final_vs_a0
  if "$PY" "$WORK/experience_docx/tools/gate_haze4k_dpga_v1_1_val.py" \
    --best_compare_json "$OUT_DIR/scout_eval_compare_v1_1_val_inner_best_vs_a0.json" \
    --final_compare_json "$OUT_DIR/scout_eval_compare_v1_1_val_inner_final_vs_a0.json" \
    --output "$OUT_DIR/gate_dpga_v1_1_val_inner.json"; then
    echo "gate_dpga_v1_1_val_inner_pass"
  else
    echo "gate_dpga_v1_1_val_inner_fail"
  fi
  echo "eval_dpga_v1_1_val_inner_done $(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG"

{
  echo "v1_1_val_inner_eval_dir=$OUT_DIR"
  echo "v1_1_val_inner_gate=$OUT_DIR/gate_dpga_v1_1_val_inner.json"
} | tee -a "$STATUS"
