#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603"
STATUS="$LOG_DIR/status.txt"

SELECTOR_TAG=${SELECTOR_TAG:-apdr_v0_2rc_frozen_selector_seed3407}
MODEL_NAME=${MODEL_NAME:-ConvIR-Haze4K-APDR-v0.2RC-frozen-selector-residual-stop20-seed3407-20260603}
EVAL_TAG=${EVAL_TAG:-apdr_v0_2rc_frozen_residual_stop20_seed3407_vs_a0}

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

log_status "start_frozen_residual root=$ROOT selector_tag=$SELECTOR_TAG model=$MODEL_NAME"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_apdr_v0_2rc_budget.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output_dir "$LOG_DIR" \
  --tag "$SELECTOR_TAG" \
  --device cuda \
  --seed 3407 \
  --global_epochs 5 \
  --spatial_epochs 3 \
  --global_batch_size 4 \
  --spatial_batch_size 8 \
  --global_resize 384 \
  --num_worker 8 \
  --global_learning_rate 2e-4 \
  --spatial_learning_rate 2e-4 \
  --global_train_batches_per_epoch 0 \
  --spatial_train_batches_per_epoch 0 \
  --calibration_images 0 \
  --budget_calibration_images 0 \
  --loss_eval_images 256 \
  --pixel_samples_per_image 2048 \
  --progress_freq 100 \
  --run_oracle_on_replay_fail \
  --save_selector_checkpoint \
  > "$LOG_DIR/selector_preflight_${SELECTOR_TAG}.log" 2>&1

SELECTOR="$LOG_DIR/selector_checkpoint_${SELECTOR_TAG}.pkl"
GATE_JSON="$LOG_DIR/gate_${SELECTOR_TAG}.json"
SUMMARY_JSON="$LOG_DIR/budget_summary_${SELECTOR_TAG}.json"

"$PY" - "$GATE_JSON" "$SUMMARY_JSON" "$SELECTOR" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selector = Path(sys.argv[3])
checks = gate["replay_gate"]["checks"]
required_action_checks = [
    "selected_train_constraints",
    "zero_residual_output",
    "auc_hard_vs_easy_by_z_img",
    "spearman_z_img_vs_a0_psnr",
    "hard_bottom25_mean_b_cons",
    "easy_top25_mean_b_cons",
    "strong_reference_mean_b_cons",
    "hard_easy_b_cons_ratio",
]
failed = [name for name in required_action_checks if not checks[name]["pass"]]
if failed:
    raise SystemExit(f"action_gate_failed: {failed}")
if not gate["oracle_gate"] or not gate["oracle_gate"]["pass"]:
    raise SystemExit("oracle_gate_failed")
if not selector.is_file():
    raise SystemExit(f"missing_selector_checkpoint: {selector}")
budget = summary.get("selected_budget_calibration", {})
if not budget.get("supported_for_checkpoint"):
    raise SystemExit(f"unsupported_budget_checkpoint: {budget}")
print("frozen_residual_precheck_pass")
PY

BEST="$ITS_ROOT/results/$MODEL_NAME/Training-Results/Best.pkl"
if [[ -f "$BEST" ]]; then
  log_status "skip_existing_train $MODEL_NAME"
else
  log_status "train_start $MODEL_NAME"
  "$PY" main.py \
    --mode train \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --mod_stats_freq 1 \
    --mod_stats_batches 64 \
    --seed 3407 \
    --init_model "$SELECTOR" \
    --arch apdr \
    --apdr_selector_mode v0_2r \
    --apdr_train_scope apdr_residual_only \
    --apdr_active_scales full \
    --apdr_loss_scales full_only \
    --apdr_prior_mode rgb_haze \
    --apdr_residual_max 0.04 \
    --apdr_gate_max 1.0 \
    --apdr_gate_init 0.01 \
    --apdr_anchor_lambda 0.10 \
    --apdr_gate_supervision_lambda 0.0 \
    --apdr_gate_lambda 0.0 \
    --apdr_residual_lambda 0.02 \
    --apdr_risk_temperature 5.0 \
    > "$LOG_DIR/train_${MODEL_NAME}.log" 2>&1
  log_status "train_done $MODEL_NAME"
fi

if [[ ! -f "$BEST" ]]; then
  log_status "missing_best_checkpoint $BEST"
  exit 1
fi

log_status "eval_start $EVAL_TAG"
"$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
  --data_dir "$DATA_DIR" \
  --original_checkpoint "$PRETRAIN" \
  --original_arch convir \
  --original_mode original \
  --original_name a0 \
  --candidate_checkpoint "$BEST" \
  --candidate_arch apdr \
  --candidate_mode original \
  --candidate_name apdr_v0_2rc_frozen_residual \
  --candidate_apdr_prior_mode rgb_haze \
  --candidate_apdr_residual_max 0.04 \
  --candidate_apdr_gate_max 1.0 \
  --candidate_apdr_gate_init 0.01 \
  --candidate_apdr_active_scales full \
  --candidate_apdr_selector_mode v0_2r \
  --output_dir "$LOG_DIR" \
  --tag "$EVAL_TAG"

"$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
  --csv "$LOG_DIR/scout_eval_per_image_${EVAL_TAG}.csv" \
  --candidate_name apdr_v0_2rc_frozen_residual \
  --output "$LOG_DIR/scout_eval_bucket_analysis_${EVAL_TAG}.json"

if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_apdr_stop20.py" \
  --compare_json "$LOG_DIR/scout_eval_compare_${EVAL_TAG}.json" \
  --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${EVAL_TAG}.json" \
  --output "$LOG_DIR/gate_${EVAL_TAG}.json" \
  --stage "APDR-v0.2RC frozen-selector residual stop20"; then
  log_status "gate_pass $EVAL_TAG"
else
  log_status "gate_fail_stop $EVAL_TAG"
  exit 0
fi

log_status "complete_frozen_residual $EVAL_TAG"
