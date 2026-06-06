#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-8-execution-queue}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
EVID=$WORK/experience_docx/experiment_logs/haze4k_v18_execution_queue_20260606
STATUS=$EVID/status.txt
SEEDS=${SEEDS:-"1701 2222 3141 4242 5151"}
DEFER_REPAIR_SEEDS=${DEFER_REPAIR_SEEDS:-"3407 2026"}

mkdir -p "$EVID"

log_status() {
  printf '%s %s\n' "$*" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
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

seed_evidence_complete() {
  local seed=$1
  local seed_evid=$EVID/seed_$seed
  local eval_out=$seed_evid/eval_regular_hard
  local selection_json=$seed_evid/v18_seed${seed}_multimetric_checkpoint_selection.json
  local label
  local split
  local label_lower

  [ -f "$selection_json" ] || return 1
  for label in model_5 model_10 model_15 model_20 Best Final; do
    label_lower=$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')
    for split in val_regular val_hard; do
      [ -f "$eval_out/scout_eval_compare_v18_seed${seed}_${label_lower}_${split}_vs_a0.json" ] || return 1
    done
  done
  return 0
}

{
  echo "v18_queue_resume_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_QUEUE_RESUME"
  echo "resume_seeds=$SEEDS"
  echo "defer_repair_seeds=$DEFER_REPAIR_SEEDS"
  echo "locked_test_touched=NO"
} | tee -a "$STATUS"

cd "$WORK"

for seed in $SEEDS; do
  SEED_EVID=$EVID/seed_$seed
  MODEL_NAME=ConvIR-Haze4K-v1.8-BiDPFM1-fusion-neighbor-seed${seed}-20260606
  MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
  TRAIN_LOG=$SEED_EVID/train_${MODEL_NAME}.log
  RESUME_TRAIN_LOG=$SEED_EVID/train_resume_${MODEL_NAME}.log
  EVAL_OUT=$SEED_EVID/eval_regular_hard
  mkdir -p "$SEED_EVID" "$EVAL_OUT"

  if seed_evidence_complete "$seed"; then
    log_status "seed_skip_complete seed=$seed model=$MODEL_NAME"
    continue
  fi

  if [ -f "$MODEL_DIR/Final.pkl" ]; then
    log_status "seed_train_skip seed=$seed reason=Final_exists model=$MODEL_NAME"
  else
    TRAIN_ARGS=(
      --model_name "$MODEL_NAME"
      --data Haze4K
      --version base
      --fam_mode original
      --arch dpga
      --dpga_fusion_mode udp_bi
      --dpga_udp_components all
      --dpga_active_adapters dpfm1
      --dpga_depth_cache_dir "$DEPTH"
      --dpga_train_depth_split train
      --dpga_eval_depth_split train
      --dpga_train_split_json "$SPLIT_JSON"
      --dpga_train_split_name train_inner
      --dpga_valid_split_json "$SPLIT_JSON"
      --dpga_valid_split_name val_regular
      --dpga_train_scope fusion_neighbor
      --dpga_neighbor_learning_rate 0.00001
      --dpga_scale_multiplier 1.0
      --dpga_adapter_residual_scale 0.1
      --dpga_tc_rec_loss charbonnier
      --dpga_tc_fft_lambda 0.05
      --dpga_tc_anchor_lambda 0.05
      --dpga_tc_chroma_lambda 0.02
      --dpga_tc_delta_lambda 0.0
      --dpga_tc_delta_tv_lambda 0.0
      --dpga_fusion_delta_lambda 0.0001
      --dpga_tc_anchor_error_threshold 0.035
      --dpga_tc_mask_mode hard_selective
      --dpga_hard_sample_lambda 0.0
      --dpga_hard_region_lambda 0.0
      --dpga_require_hard_labels 1
      --dpga_hard_sampler_json "$SPLIT_JSON"
      --dpga_hard_sampler_split_name train_inner
      --dpga_hard_sampler_seed "$seed"
      --dpga_hard_sampler_hard_ratio 0.3333333333
      --dpga_hard_sampler_medium_ratio 0.3333333333
      --mode train
      --data_dir "$DATA"
      --batch_size 8
      --leaning_rate 0.0001
      --weight_decay 0.0001
      --grad_clip_norm 0.001
      --num_epoch 1000
      --stop_epoch 20
      --print_freq 50
      --num_worker 8
      --save_freq 5
      --valid_freq 1
      --mod_stats_freq 1
      --mod_stats_batches 64
      --seed "$seed"
    )

    train_log_path=$TRAIN_LOG
    train_mode=start
    if [ -f "$MODEL_DIR/model.pkl" ]; then
      TRAIN_ARGS+=(--resume "$MODEL_DIR/model.pkl")
      train_log_path=$RESUME_TRAIN_LOG
      train_mode=resume
      log_status "seed_train_start seed=$seed model=$MODEL_NAME mode=resume resume=$MODEL_DIR/model.pkl log=$train_log_path"
    else
      TRAIN_ARGS+=(--init_model "$A0")
      log_status "seed_train_start seed=$seed model=$MODEL_NAME mode=fresh log=$train_log_path"
    fi

    set +e
    (
      cd "$WORK/Dehazing/ITS"
      PYTHONUNBUFFERED=1 "$PY" main.py "${TRAIN_ARGS[@]}"
    ) > "$train_log_path" 2>&1
    train_rc=$?
    set -e
    log_status "seed_train_done seed=$seed rc=$train_rc model=$MODEL_NAME mode=$train_mode log=$train_log_path"
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
      json_path="$EVAL_OUT/scout_eval_compare_v18_seed${seed}_${label_lower}_${split}_vs_a0.json"
      if [ -f "$json_path" ]; then
        log_status "step_skip name=seed_${seed}_${label}_${split}_eval reason=compare_exists path=$json_path"
        continue
      fi
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
  fi
done

log_status "resume_queue_defers_repair seeds=$DEFER_REPAIR_SEEDS"
log_status "state=COMPLETED_QUEUE_PENDING_REPAIR"
printf 'V18_EXECUTION_QUEUE_RESUME_OK\n'
