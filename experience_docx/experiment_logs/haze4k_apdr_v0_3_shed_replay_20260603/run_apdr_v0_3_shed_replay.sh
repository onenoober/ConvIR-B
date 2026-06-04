#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-3-shed-diagnostics}
APDR_ITS_ROOT="$ROOT/Dehazing/ITS"
PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
SOURCE_APDR_ROOT=${SOURCE_APDR_ROOT:-/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic}
SELECTOR=${SELECTOR:-$SOURCE_APDR_ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl}
LOG_DIR="$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_3_shed_replay_20260603"
STATUS="$LOG_DIR/status.txt"

mkdir -p "$LOG_DIR"
cd "$ROOT"

log_status() {
  echo "$* $(date --iso-8601=seconds)" | tee -a "$STATUS"
}

run_candidate() {
  local name="$1"
  local expert_root="$2"
  local checkpoint="$3"
  local arch="$4"
  local mode="$5"
  local scm_mode="$6"
  shift 6
  local tag="apdr_v0_3_shed_${name}_seed3407_vs_a0"
  local candidate_name="safe_expert_${name}"

  log_status "replay_start $name"
  if [[ ! -f "$checkpoint" ]]; then
    log_status "missing_checkpoint $name $checkpoint"
    return 0
  fi

  if ! "$PY" "$ROOT/experience_docx/tools/replay_haze4k_apdr_safe_expert_delta.py" \
    --data_dir "$DATA_DIR" \
    --apdr_its_root "$APDR_ITS_ROOT" \
    --apdr_selector_checkpoint "$SELECTOR" \
    --apdr_selector_mode v0_2r \
    --apdr_active_scales full \
    --apdr_gate_max 1.0 \
    --apdr_gate_init 0.01 \
    --residual_max 0.04 \
    --expert_name "$name" \
    --expert_its_root "$expert_root/Dehazing/ITS" \
    --expert_checkpoint "$checkpoint" \
    --expert_arch "$arch" \
    --expert_mode "$mode" \
    --expert_scm_mode "$scm_mode" \
    --output_dir "$LOG_DIR" \
    --tag "$tag" \
    --device cuda \
    --progress_freq 100 \
    "$@" \
    > "$LOG_DIR/replay_${tag}.log" 2>&1; then
    log_status "replay_fail $name"
    return 0
  fi

  if ! "$PY" "$ROOT/experience_docx/tools/analyze_haze4k_delta_buckets.py" \
    --csv "$LOG_DIR/scout_eval_per_image_${tag}.csv" \
    --candidate_name "$candidate_name" \
    --output "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json"; then
    log_status "bucket_analysis_fail $name"
    return 0
  fi

  if "$PY" "$ROOT/experience_docx/tools/gate_haze4k_apdr_stop20.py" \
    --compare_json "$LOG_DIR/scout_eval_compare_${tag}.json" \
    --bucket_json "$LOG_DIR/scout_eval_bucket_analysis_${tag}.json" \
    --output "$LOG_DIR/gate_${tag}.json" \
    --stage "APDR-v0.3 SHED safe expert replay $name"; then
    log_status "gate_pass $name"
  else
    log_status "gate_fail $name"
  fi
}

log_status "start root=$ROOT selector=$SELECTOR"

run_candidate \
  fam2_only \
  /root/autodl-tmp/workspace/ConvIR-B \
  /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam2_modres-stop20-seed3407-20260531/Training-Results/Best.pkl \
  convir \
  fam2_modres \
  original

run_candidate \
  fam2_conf_gated \
  /root/autodl-tmp/workspace/ConvIR-B \
  /root/autodl-tmp/workspace/ConvIR-B/Dehazing/ITS/results/ConvIR-Haze4K-fam2_modres_gamma_conf_gated-stop20-seed3407-20260601/Training-Results/Best.pkl \
  convir \
  fam2_modres_gamma_conf_gated \
  original

run_candidate \
  hardfreq \
  /root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss \
  /root/autodl-tmp/workspace/ConvIR-B-hardfreq-loss/Dehazing/ITS/results/ConvIR-Haze4K-hardfreq-loss-stop20-seed3407-20260601/Training-Results/Best.pkl \
  convir \
  original \
  original

run_candidate \
  pfd_b1 \
  /root/autodl-tmp/workspace/ConvIR-B-pfd-mainline \
  /root/autodl-tmp/workspace/ConvIR-B-pfd-mainline/Dehazing/ITS/results/ConvIR-Haze4K-PFD-B1-rhfd-stop20-seed3407-20260602/Training-Results/Best.pkl \
  pfd \
  original \
  original \
  --expert_pfd_rhfd 1 \
  --expert_pfd_hscm 0 \
  --expert_pfd_pffb 0

run_candidate \
  haze_prior_scm \
  /root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm \
  /root/autodl-tmp/workspace/ConvIR-B-haze-prior-scm/Dehazing/ITS/results/ConvIR-Haze4K-hazeprior-scm-hardaux-stop20-seed3407-20260601/Training-Results/Best.pkl \
  convir \
  original \
  haze_prior

log_status "complete"
