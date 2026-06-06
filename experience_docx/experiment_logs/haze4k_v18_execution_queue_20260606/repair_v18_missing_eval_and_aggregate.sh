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
SEEDS=${SEEDS:-"3407 2026 929 123 777 1701 2222 3141 4242 5151"}
REPAIR_OUT=$EVID/v18_eval_repair
REPAIR_LOG=$REPAIR_OUT/repair_v18_missing_eval_and_aggregate.log

mkdir -p "$REPAIR_OUT"

log_status() {
  printf '%s %s\n' "$*" "$(date --iso-8601=seconds)" | tee -a "$STATUS"
}

run_logged() {
  local step_name=$1
  local log_path=$2
  shift 2
  mkdir -p "$(dirname "$log_path")"
  log_status "repair_step_start name=$step_name log=$log_path"
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e
  log_status "repair_step_done name=$step_name rc=$rc log=$log_path"
  return 0
}

needs_repair() {
  local seed=$1
  local selection_json=$EVID/seed_$seed/v18_seed${seed}_multimetric_checkpoint_selection.json
  if [ ! -f "$selection_json" ]; then
    return 0
  fi
  "$PY" - "$selection_json" <<'PY'
import json
import sys
path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if payload.get("selected_checkpoint_label") is None:
    raise SystemExit(0)
rows = payload.get("rows") or []
if any(row.get("decision") == "MISSING_COMPARE_JSON" for row in rows):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

{
  printf 'repair_wait_start %s\n' "$(date --iso-8601=seconds)"
  while tmux has-session -t v18_execution_queue 2>/dev/null; do
    printf 'main_queue_still_active %s\n' "$(date --iso-8601=seconds)"
    sleep 300
  done
  printf 'main_queue_inactive_start_repair %s\n' "$(date --iso-8601=seconds)"
} | tee -a "$REPAIR_LOG"

log_status "repair_start name=v18_missing_eval_and_aggregate"
selection_jsons=()
repaired_seeds=()
for seed in $SEEDS; do
  SEED_EVID=$EVID/seed_$seed
  MODEL_NAME=ConvIR-Haze4K-v1.8-BiDPFM1-fusion-neighbor-seed${seed}-20260606
  MODEL_DIR=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results
  EVAL_OUT=$SEED_EVID/eval_regular_hard
  mkdir -p "$EVAL_OUT"

  if ! needs_repair "$seed"; then
    if [ -f "$SEED_EVID/v18_seed${seed}_multimetric_checkpoint_selection.json" ]; then
      selection_jsons+=("$SEED_EVID/v18_seed${seed}_multimetric_checkpoint_selection.json")
    fi
    continue
  fi

  repaired_seeds+=("$seed")
  log_status "repair_seed_start seed=$seed reason=missing_or_empty_compare_json"
  checkpoints=()
  for label in model_5 model_10 model_15 model_20 Best Final; do
    if [ -f "$MODEL_DIR/$label.pkl" ]; then
      checkpoints+=("$label")
    else
      log_status "repair_checkpoint_missing seed=$seed label=$label path=$MODEL_DIR/$label.pkl"
    fi
  done

  for label in "${checkpoints[@]}"; do
    label_lower=$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')
    for split in val_regular val_hard; do
      log_path="$EVAL_OUT/repair_eval_v18_seed${seed}_${label_lower}_${split}.log"
      run_logged "repair_seed_${seed}_${label}_${split}_eval" "$log_path" \
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
    run_logged "repair_seed_${seed}_multimetric_checkpoint_selection" "$SEED_EVID/repair_v18_seed${seed}_checkpoint_selection.log" \
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
  log_status "repair_seed_done seed=$seed"
done

AGG_OUT=$EVID/v18_multiseed_aggregate
mkdir -p "$AGG_OUT"
if [ "${#selection_jsons[@]}" -gt 0 ]; then
  run_logged "repair_v18_multiseed_aggregate" "$AGG_OUT/repair_v18_multiseed_aggregate.log" \
    "$PY" experience_docx/tools/aggregate_haze4k_multiseed_checkpoint_metrics.py \
      --selection_jsons "${selection_jsons[@]}" \
      --output_json "$AGG_OUT/v18_multiseed_aggregate_summary.json" \
      --output_csv "$AGG_OUT/v18_multiseed_aggregate_metrics.csv"
else
  log_status "repair_v18_multiseed_aggregate_skipped reason=no_selection_jsons"
fi

printf '%s\n' "${repaired_seeds[@]}" > "$REPAIR_OUT/repaired_seeds.txt"
log_status "repair_done name=v18_missing_eval_and_aggregate repaired_seed_count=${#repaired_seeds[@]}"
printf 'V18_EVAL_REPAIR_OK\n'
