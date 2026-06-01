#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-pfd-mainline}
ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
PRETRAIN=${PRETRAIN:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_pfd_mainline_20260602"
STATUS="$LOG_DIR/status.txt"

A1_MODEL=ConvIR-Haze4K-A1-init-official-stop20-seed3407-20260602
B1_MODEL=ConvIR-Haze4K-PFD-B1-rhfd-stop20-seed3407-20260602
B2_MODEL=ConvIR-Haze4K-PFD-B2-rhfd-hscm-lite-stop20-seed3407-20260602
B3_MODEL=ConvIR-Haze4K-PFD-B3-rhfd-hscm-pffb-low-stop20-seed3407-20260602

mkdir -p "$LOG_DIR"
cd "$ITS_ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

run_train() {
  local model_name="$1"
  local log_name="$2"
  shift 2
  local best_path="$ITS_ROOT/results/$model_name/Training-Results/Best.pkl"
  if [[ -f "$best_path" ]]; then
    log_status "skip_existing $model_name"
    return
  fi
  log_status "train_start $model_name"
  "$PY" main.py \
    --mode train \
    --model_name "$model_name" \
    --data Haze4K \
    --data_dir "$DATA_DIR" \
    --version base \
    --batch_size 8 \
    --learning_rate 4e-4 \
    --num_epoch 1000 \
    --stop_epoch 20 \
    --print_freq 50 \
    --num_worker 8 \
    --save_freq 5 \
    --valid_freq 1 \
    --seed 3407 \
    --init_model "$PRETRAIN" \
    "$@" > "$LOG_DIR/$log_name" 2>&1
  log_status "train_done $model_name rc=$?"
}

compare_and_gate() {
  local stage="$1"
  local tag="$2"
  local original_checkpoint="$3"
  local candidate_checkpoint="$4"
  local original_name="$5"
  local candidate_name="$6"
  shift 6

  log_status "eval_start $stage"
  "$PY" "$ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
    --data_dir "$DATA_DIR" \
    --original_checkpoint "$original_checkpoint" \
    --original_name "$original_name" \
    --candidate_checkpoint "$candidate_checkpoint" \
    --candidate_name "$candidate_name" \
    --output_dir "$LOG_DIR" \
    --tag "$tag" \
    "$@"

  "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_${tag}.csv" \
    --candidate_name "$candidate_name" \
    --output "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json"

  if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_pfd_stop20.py" \
    --stage "$stage" \
    --compare_json "$LOG_DIR/scout_eval_compare_${tag}.json" \
    --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json" \
    --output "$LOG_DIR/gate_${stage}_stop20.json"; then
    log_status "gate_pass $stage"
  else
    log_status "gate_fail_stop $stage"
    exit 0
  fi
}

log_status "start"

"$PY" "$ROOT/experience_docx/tools/preflight_haze4k_pfd.py" \
  --data_dir "$DATA_DIR" \
  --checkpoint "$PRETRAIN" \
  --output "$LOG_DIR/preflight_pfd_v0.json" \
  --device cuda
log_status "preflight_pass"

run_train "$A1_MODEL" "A1_stop20_seed3407.log" --arch convir
run_train "$B1_MODEL" "B1_rhfd_stop20_seed3407.log" \
  --arch pfd --pfd_rhfd 1 --pfd_hscm 0 --pfd_pffb 0 --pfd_teacher 0

A1_BEST="$ITS_ROOT/results/$A1_MODEL/Training-Results/Best.pkl"
B1_BEST="$ITS_ROOT/results/$B1_MODEL/Training-Results/Best.pkl"
B2_BEST="$ITS_ROOT/results/$B2_MODEL/Training-Results/Best.pkl"
B3_BEST="$ITS_ROOT/results/$B3_MODEL/Training-Results/Best.pkl"

compare_and_gate B1 seed3407_B1_vs_A1_best "$A1_BEST" "$B1_BEST" a1 b1 \
  --original_arch convir --original_mode original \
  --candidate_arch pfd --candidate_mode original \
  --candidate_pfd_rhfd 1 --candidate_pfd_hscm 0 --candidate_pfd_pffb 0

run_train "$B2_MODEL" "B2_rhfd_hscm_lite_stop20_seed3407.log" \
  --arch pfd --pfd_rhfd 1 --pfd_hscm 1 --pfd_pffb 0 --pfd_teacher 0

compare_and_gate B2 seed3407_B2_vs_B1_best "$B1_BEST" "$B2_BEST" b1 b2 \
  --original_arch pfd --original_mode original \
  --original_pfd_rhfd 1 --original_pfd_hscm 0 --original_pfd_pffb 0 \
  --candidate_arch pfd --candidate_mode original \
  --candidate_pfd_rhfd 1 --candidate_pfd_hscm 1 --candidate_pfd_pffb 0

run_train "$B3_MODEL" "B3_rhfd_hscm_pffb_low_stop20_seed3407.log" \
  --arch pfd --pfd_rhfd 1 --pfd_hscm 1 --pfd_pffb 1 --pfd_pffb_high 0 --pfd_teacher 0

compare_and_gate B3 seed3407_B3_vs_B2_best "$B2_BEST" "$B3_BEST" b2 b3 \
  --original_arch pfd --original_mode original \
  --original_pfd_rhfd 1 --original_pfd_hscm 1 --original_pfd_pffb 0 \
  --candidate_arch pfd --candidate_mode original \
  --candidate_pfd_rhfd 1 --candidate_pfd_hscm 1 --candidate_pfd_pffb 1

log_status "complete"
