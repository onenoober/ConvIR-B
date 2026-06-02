#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-b1r-decoder-rhfd-preserve}
ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602"
STATUS="$LOG_DIR/status.txt"

B1R10_MODEL=ConvIR-Haze4K-B1r-decoder-rhfd-adapter-only-stop10-seed3407-20260602
B1R20_MODEL=ConvIR-Haze4K-B1r-decoder-rhfd-adapter-only-stop20-seed3407-20260602
RUN_STOP20=${RUN_STOP20:-1}

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

run_b1r_train() {
  local model_name="$1"
  local stop_epoch="$2"
  local log_name="$3"
  local best_path="$ITS_ROOT/results/$model_name/Training-Results/Best.pkl"
  if [[ -f "$best_path" ]]; then
    log_status "skip_existing $model_name"
    return
  fi
  log_status "train_start $model_name stop_epoch=$stop_epoch"
  "$PY" main.py \
    --mode train \
    --model_name "$model_name" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epoch 1000 \
    --stop_epoch "$stop_epoch" \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    --init_model "$PRETRAIN" \
    --arch pfd \
    --pfd_rhfd 0 \
    --pfd_hscm 0 \
    --pfd_pffb 0 \
    --pfd_pffb_high 0 \
    --pfd_teacher 0 \
    --pfd_decoder_rhfd 1 \
    --pfd_decoder_rhfd_scale 0.1 \
    --pfd_adapter_only 1 \
    > "$LOG_DIR/$log_name" 2>&1
  log_status "train_done $model_name rc=$?"
}

compare_b1r() {
  local stage="$1"
  local tag="$2"
  local candidate_checkpoint="$3"

  log_status "eval_start $stage"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$PRETRAIN" \
    --original_arch convir \
    --original_mode original \
    --original_name a0 \
    --candidate_checkpoint "$candidate_checkpoint" \
    --candidate_arch pfd \
    --candidate_mode original \
    --candidate_name b1r \
    --candidate_pfd_rhfd 0 \
    --candidate_pfd_hscm 0 \
    --candidate_pfd_pffb 0 \
    --candidate_pfd_pffb_high 0 \
    --candidate_pfd_teacher 0 \
    --candidate_pfd_decoder_rhfd 1 \
    --candidate_pfd_decoder_rhfd_scale 0.1 \
    --output_dir "$LOG_DIR" \
    --tag "$tag"

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_${tag}.csv" \
    --candidate_name b1r \
    --output "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json"
  log_status "eval_done $stage"
}

gate_b1r_stop20() {
  local tag="$1"
  if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_pfd_stop20.py" \
    --stage B1r \
    --compare_json "$LOG_DIR/scout_eval_compare_${tag}.json" \
    --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json" \
    --output "$LOG_DIR/gate_B1r_stop20.json"; then
    log_status "gate_pass B1r"
  else
    log_status "gate_fail_stop B1r"
    exit 0
  fi
}

log_status "start"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_pfd.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output "$LOG_DIR/preflight_b1r_decoder_rhfd.json" \
  --device cuda \
  --pfd_decoder_rhfd 1 \
  --pfd_decoder_rhfd_scale 0.1
log_status "preflight_pass"

run_b1r_train "$B1R10_MODEL" 10 "B1r_adapter_only_stop10_seed3407.log"
B1R10_BEST="$ITS_ROOT/results/$B1R10_MODEL/Training-Results/Best.pkl"
compare_b1r B1r_stop10 seed3407_B1r_stop10_vs_A0_best "$B1R10_BEST"

if [[ "$RUN_STOP20" != "1" ]]; then
  log_status "stop_after_stop10 RUN_STOP20=$RUN_STOP20"
  exit 0
fi

run_b1r_train "$B1R20_MODEL" 20 "B1r_adapter_only_stop20_seed3407.log"
B1R20_BEST="$ITS_ROOT/results/$B1R20_MODEL/Training-Results/Best.pkl"
compare_b1r B1r_stop20 seed3407_B1r_stop20_vs_A0_best "$B1R20_BEST"
gate_b1r_stop20 seed3407_B1r_stop20_vs_A0_best

log_status "complete"
