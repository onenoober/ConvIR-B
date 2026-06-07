#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
V17_FEATURE_CSV=${V17_FEATURE_CSV:-$WORK/experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_fulltrain_features/v17_fulltrain_a0_udp_feature_table.csv}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
EVID=$WORK/experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606
STATUS=$EVID/status.txt
SEEDS=${SEEDS:-"3407 2026 929 123 777 1701 2222 3141 4242 5151"}

mkdir -p "$EVID"

log_status() {
  printf '%s %s\n' "$*" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
}

run_step() {
  local step_name=$1
  shift
  log_status "step_start name=$step_name"
  set +e
  "$@"
  local rc=$?
  set -e
  log_status "step_done name=$step_name rc=$rc"
  return 0
}

run_logged() {
  local step_name=$1
  local log_path=$2
  shift 2
  mkdir -p "$(dirname "$log_path")"
  log_status "step_start name=$step_name log=$log_path"
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e
  log_status "step_done name=$step_name rc=$rc log=$log_path"
  return 0
}

{
  echo "v18_queue_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_QUEUE"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "split_json=$SPLIT_JSON"
  echo "v17_feature_csv=$V17_FEATURE_CSV"
  echo "seeds=$SEEDS"
  echo "locked_test_touched=NO"
  if [ -d "$WORK/.git" ]; then
    branch=$(git -C "$WORK" branch --show-current 2>/dev/null || true)
    commit=$(git -C "$WORK" rev-parse --short HEAD 2>/dev/null || true)
    printf 'branch=%s\n' "${branch:-UNKNOWN_OR_UNSAFE_GIT_DIR}"
    printf 'commit=%s\n' "${commit:-UNKNOWN_OR_UNSAFE_GIT_DIR}"
    git -C "$WORK" status --short 2>/dev/null | sed 's/^/git_status=/' || echo "git_status=UNREADABLE_OR_UNSAFE_GIT_DIR"
  else
    echo "git_status=NO_GIT_COPIED_WORKTREE"
  fi
} | tee -a "$STATUS"

cd "$WORK"

ROUTER_OUT=$EVID/v18_router_policy
run_logged "v18_router_policy_table_analysis" "$ROUTER_OUT/v18_router_policy.log" \
  "$PY" experience_docx/tools/analyze_haze4k_v18_router_policy.py \
    --feature_csv "$V17_FEATURE_CSV" \
    --output_dir "$ROUTER_OUT"

DOMAIN_OUT=$EVID/v18_domain_data_preflight
run_logged "v18_domain_data_preflight" "$DOMAIN_OUT/v18_domain_data_preflight.log" \
  "$PY" experience_docx/tools/audit_haze4k_v18_domain_data_preflight.py \
    --data_dir "$DATA" \
    --split_json "$SPLIT_JSON" \
    --feature_csv "$V17_FEATURE_CSV" \
    --splits train_inner val_regular val_hard \
    --output_dir "$DOMAIN_OUT"

DOMAIN_ADAPT_OUT=$EVID/v18_domain_adaptation_q5
run_logged "v18_domain_adaptation_q5" "$DOMAIN_ADAPT_OUT/v18_domain_adaptation_q5.log" \
  "$PY" experience_docx/tools/analyze_haze4k_v18_domain_adaptation.py \
    --feature_csv "$V17_FEATURE_CSV" \
    --domain_csv "$DOMAIN_OUT/v18_domain_data_preflight_per_image.csv" \
    --output_dir "$DOMAIN_ADAPT_OUT" \
    --real_data_candidates \
      "$WORK/Dehazing/ITS/datasets/real_haze" \
      "$WORK/dataset/real_haze" \
      "$WORK/datasets/real_haze" \
      /root/autodl-tmp/workspace/Dehaze-Net/dataset/real_haze \
      /root/autodl-tmp/workspace/dataset/real_haze \
      /root/autodl-tmp/workspace/datasets/real_haze \
      /root/autodl-tmp/dataset/real_haze \
      /root/autodl-tmp/datasets/real_haze

selection_jsons=()
for seed in $SEEDS; do
  SEED_EVID=$EVID/seed_$seed
  MODEL_NAME=ConvIR-Haze4K-v1.8-BiDPFM1-fusion-neighbor-seed${seed}-20260606
  MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
  TRAIN_LOG=$SEED_EVID/train_${MODEL_NAME}.log
  EVAL_OUT=$SEED_EVID/eval_regular_hard
  mkdir -p "$SEED_EVID" "$EVAL_OUT"

  if [ -f "$MODEL_DIR/Final.pkl" ]; then
    log_status "seed_train_skip seed=$seed reason=Final_exists model=$MODEL_NAME"
  else
    log_status "seed_train_start seed=$seed model=$MODEL_NAME"
    set +e
    (
      cd "$WORK/Dehazing/ITS"
      PYTHONUNBUFFERED=1 "$PY" main.py \
        --model_name "$MODEL_NAME" \
        --data Haze4K \
        --version base \
        --fam_mode original \
        --arch dpga \
        --dpga_fusion_mode udp_bi \
        --dpga_udp_components all \
        --dpga_active_adapters dpfm1 \
        --dpga_depth_cache_dir "$DEPTH" \
        --dpga_train_depth_split train \
        --dpga_eval_depth_split train \
        --dpga_train_split_json "$SPLIT_JSON" \
        --dpga_train_split_name train_inner \
        --dpga_valid_split_json "$SPLIT_JSON" \
        --dpga_valid_split_name val_regular \
        --dpga_train_scope fusion_neighbor \
        --dpga_neighbor_learning_rate 0.00001 \
        --dpga_scale_multiplier 1.0 \
        --dpga_adapter_residual_scale 0.1 \
        --dpga_tc_rec_loss charbonnier \
        --dpga_tc_fft_lambda 0.05 \
        --dpga_tc_anchor_lambda 0.05 \
        --dpga_tc_chroma_lambda 0.02 \
        --dpga_tc_delta_lambda 0.0 \
        --dpga_tc_delta_tv_lambda 0.0 \
        --dpga_fusion_delta_lambda 0.0001 \
        --dpga_tc_anchor_error_threshold 0.035 \
        --dpga_tc_mask_mode hard_selective \
        --dpga_hard_sample_lambda 0.0 \
        --dpga_hard_region_lambda 0.0 \
        --dpga_require_hard_labels 1 \
        --dpga_hard_sampler_json "$SPLIT_JSON" \
        --dpga_hard_sampler_split_name train_inner \
        --dpga_hard_sampler_seed "$seed" \
        --dpga_hard_sampler_hard_ratio 0.3333333333 \
        --dpga_hard_sampler_medium_ratio 0.3333333333 \
        --mode train \
        --data_dir "$DATA" \
        --batch_size 8 \
        --leaning_rate 0.0001 \
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
        --seed "$seed"
    ) > "$TRAIN_LOG" 2>&1
    train_rc=$?
    set -e
    log_status "seed_train_done seed=$seed rc=$train_rc model=$MODEL_NAME log=$TRAIN_LOG"
  fi

  checkpoints=()
  for label in model_5 model_10 model_15 model_20 Best Final; do
    if [ -f "$MODEL_DIR/$label.pkl" ]; then
      checkpoints+=("$label")
    else
      log_status "checkpoint_missing seed=$seed label=$label path=$MODEL_DIR/$label.pkl"
    fi
  done

  for label in "${checkpoints[@]}"; do
    label_lower=$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')
    for split in val_regular val_hard; do
      log_path="$EVAL_OUT/eval_v18_seed${seed}_${label_lower}_${split}.log"
      run_logged "seed_${seed}_${label}_${split}_eval" "$log_path" \
        "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
          --data_dir "$DATA" \
          --original_checkpoint "$A0" \
          --original_arch convir \
          --original_mode original \
          --original_name a0 \
          --candidate_checkpoint "$MODEL_DIR/$label.pkl" \
          --candidate_arch dpga \
          --candidate_mode "$label" \
          --candidate_name "v18_seed${seed}_${label_lower}_${split}" \
          --candidate_dpga_fusion_mode udp_bi \
          --candidate_dpga_active_adapters dpfm1 \
          --candidate_dpga_udp_components all \
          --candidate_dpga_scale_multiplier 1.0 \
          --dpga_depth_cache_dir "$DEPTH" \
          --split_json "$SPLIT_JSON" \
          --split_name "$split" \
          --output_dir "$EVAL_OUT" \
          --tag "v18_seed${seed}_${label_lower}_${split}_vs_a0"
    done
  done

  if [ "${#checkpoints[@]}" -gt 0 ]; then
    selection_json="$SEED_EVID/v18_seed${seed}_multimetric_checkpoint_selection.json"
    selection_csv="$SEED_EVID/v18_seed${seed}_multimetric_checkpoint_selection.csv"
    run_logged "seed_${seed}_multimetric_checkpoint_selection" "$SEED_EVID/v18_seed${seed}_checkpoint_selection.log" \
      "$PY" experience_docx/tools/select_haze4k_multimetric_checkpoint.py \
        --eval_dir "$EVAL_OUT" \
        --candidate_prefix "v18_seed${seed}" \
        --checkpoint_labels "${checkpoints[@]}" \
        --output_json "$selection_json" \
        --output_csv "$selection_csv"
    if [ -f "$selection_json" ]; then
      selection_jsons+=("$selection_json")
    fi
  fi
done

AGG_OUT=$EVID/v18_multiseed_aggregate
mkdir -p "$AGG_OUT"
if [ "${#selection_jsons[@]}" -gt 0 ]; then
  run_logged "v18_multiseed_aggregate" "$AGG_OUT/v18_multiseed_aggregate.log" \
    "$PY" experience_docx/tools/aggregate_haze4k_multiseed_checkpoint_metrics.py \
      --selection_jsons "${selection_jsons[@]}" \
      --output_json "$AGG_OUT/v18_multiseed_aggregate_summary.json" \
      --output_csv "$AGG_OUT/v18_multiseed_aggregate_metrics.csv"
else
  log_status "v18_multiseed_aggregate_skipped reason=no_selection_jsons"
fi

log_status "state=COMPLETED_QUEUE_PENDING_SYNC"
printf 'V18_EXECUTION_QUEUE_OK\n'
