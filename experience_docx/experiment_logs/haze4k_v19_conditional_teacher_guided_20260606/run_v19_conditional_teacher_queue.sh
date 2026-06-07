#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/convir-cu128/bin/python}
WORK=${WORK:-/root/autodl-tmp/workspace/ConvIR-B-v1-9-conditional-teacher}
CONVIR_ITS=${CONVIR_ITS:-$WORK/Dehazing/ITS}
UDP_REPO=${UDP_REPO:-/root/autodl-tmp/workspace/UDPNet}
DATA=${DATA:-/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K}
DEPTH=${DEPTH:-/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}
A0=${A0:-/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-/root/autodl-tmp/workspace/UDPNet_official_download/ConvIR_UDPNet_haze4k.ckpt}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dpga_v13_hsdf_20260604/internal_val/haze4k_dpga_v13_regular_hard_seed3407.json}
V17_FEATURE_CSV=${V17_FEATURE_CSV:-$WORK/experience_docx/experiment_logs/haze4k_v17_rc_expert_mix_20260605/v17_fulltrain_features/v17_fulltrain_a0_udp_feature_table.csv}
EVID=$WORK/experience_docx/experiment_logs/haze4k_v19_conditional_teacher_guided_20260606
STATUS=$EVID/status.txt
SEEDS=${SEEDS:-"3407 2026 929"}
HYGIENE=${HYGIENE:-"clip0p001_noema:0.001:0 clip0p01_noema:0.01:0 clip0p1_noema:0.1:0 clip0p01_ema:0.01:1"}

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

{
  echo "v19_queue_start $(date --iso-8601=seconds)"
  echo "state=RUNNING_QUEUE"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "depth=$DEPTH"
  echo "a0=$A0"
  echo "official_ckpt=$OFFICIAL_CKPT"
  echo "split_json=$SPLIT_JSON"
  echo "v17_feature_csv=$V17_FEATURE_CSV"
  echo "seeds=$SEEDS"
  echo "hygiene=$HYGIENE"
  echo "locked_test_touched=NO"
  if [ -d "$WORK/.git" ]; then
    git -C "$WORK" branch --show-current 2>/dev/null | sed 's/^/branch=/'
    git -C "$WORK" rev-parse --short HEAD 2>/dev/null | sed 's/^/commit=/'
    git -C "$WORK" status --short 2>/dev/null | sed 's/^/git_status=/'
  fi
} | tee -a "$STATUS"

cd "$WORK"

PHYS_OUT=$EVID/v19_physical_prior_preflight
run_logged "v19_physical_prior_preflight" "$PHYS_OUT/v19_physical_prior_preflight.log" \
  "$PY" experience_docx/tools/audit_haze4k_v19_physical_prior_preflight.py \
    --data_dir "$DATA" \
    --extra_roots /root/autodl-tmp/workspace/Dehaze-Net /root/autodl-tmp/workspace \
    --output_dir "$PHYS_OUT"

if [ ! -f "$V17_FEATURE_CSV" ]; then
  V17_OUT=$EVID/v19_rebuilt_v17_features
  run_logged "v19_rebuild_v17_feature_table" "$V17_OUT/v19_rebuild_v17_feature_table.log" \
    "$PY" experience_docx/tools/extract_haze4k_v17_fulltrain_a0_udp_features.py \
      --convir_its_dir "$CONVIR_ITS" \
      --udp_repo "$UDP_REPO" \
      --data_dir "$DATA" \
      --depth_cache_dir "$DEPTH" \
      --a0_checkpoint "$A0" \
      --official_checkpoint "$OFFICIAL_CKPT" \
      --split_json "$SPLIT_JSON" \
      --splits train_inner val_regular val_hard \
      --depth_split train \
      --output_dir "$V17_OUT"
  V17_FEATURE_CSV=$V17_OUT/v17_fulltrain_a0_udp_feature_table.csv
fi

PRED_OUT=$EVID/v19_teacher_delta_predictability
run_logged "v19_teacher_delta_predictability" "$PRED_OUT/v19_teacher_delta_predictability.log" \
  "$PY" experience_docx/tools/analyze_haze4k_v19_teacher_delta_predictability.py \
    --feature_csv "$V17_FEATURE_CSV" \
    --output_dir "$PRED_OUT"

PATCH_OUT=$EVID/v19_patch_alpha_oracle
run_logged "v19_patch_alpha_oracle" "$PATCH_OUT/v19_patch_alpha_oracle.log" \
  "$PY" experience_docx/tools/extract_haze4k_v19_patch_alpha_oracle.py \
    --convir_its_dir "$CONVIR_ITS" \
    --udp_repo "$UDP_REPO" \
    --data_dir "$DATA" \
    --depth_cache_dir "$DEPTH" \
    --a0_checkpoint "$A0" \
    --official_checkpoint "$OFFICIAL_CKPT" \
    --split_json "$SPLIT_JSON" \
    --splits train_inner val_regular val_hard \
    --depth_split train \
    --tile_size 64 \
    --output_dir "$PATCH_OUT"

MASK_OUT=$EVID/v19_patch_mask_head
if [ -f "$PATCH_OUT/v19_patch_alpha_oracle_tiles.csv" ]; then
  run_logged "v19_patch_mask_head" "$MASK_OUT/v19_patch_mask_head.log" \
    "$PY" experience_docx/tools/train_haze4k_v19_patch_mask_head.py \
      --tile_csv "$PATCH_OUT/v19_patch_alpha_oracle_tiles.csv" \
      --output_dir "$MASK_OUT"
else
  log_status "v19_patch_mask_head_skipped reason=missing_tile_csv"
fi

selection_jsons=()
for seed in $SEEDS; do
  SEED_OUT=$EVID/seed_$seed
  MODEL_NAME=ConvIR-Haze4K-v1.9-CondTeacher-seed${seed}-20260606
  mkdir -p "$SEED_OUT/eval_regular_hard"
  run_logged "v19_student_train_seed_${seed}" "$SEED_OUT/train_${MODEL_NAME}.log" \
    "$PY" experience_docx/tools/train_haze4k_v19_conditional_student.py \
      --udp_repo "$UDP_REPO" \
      --data_dir "$DATA" \
      --depth_cache_dir "$DEPTH" \
      --a0_checkpoint "$A0" \
      --official_checkpoint "$OFFICIAL_CKPT" \
      --split_json "$SPLIT_JSON" \
      --model_name "$MODEL_NAME" \
      --output_dir "$SEED_OUT" \
      --seed "$seed" \
      --epochs 20 \
      --batch_size 4 \
      --grad_clip_norm 0.01

  CKPT_DIR=$SEED_OUT/checkpoints
  for label in model_5 model_10 model_15 model_20 Final; do
    if [ ! -f "$CKPT_DIR/$label.pkl" ]; then
      log_status "v19_checkpoint_missing seed=$seed label=$label path=$CKPT_DIR/$label.pkl"
      continue
    fi
    lower=$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')
    for split in val_regular val_hard; do
      run_logged "v19_eval_seed_${seed}_${label}_${split}" "$SEED_OUT/eval_regular_hard/eval_${lower}_${split}.log" \
        "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
          --data_dir "$DATA" \
          --original_checkpoint "$A0" \
          --original_arch convir \
          --original_mode original \
          --original_name a0 \
          --candidate_checkpoint "$CKPT_DIR/$label.pkl" \
          --candidate_arch dpga \
          --candidate_mode "$label" \
          --candidate_name "v19_seed${seed}_${lower}_${split}" \
          --candidate_dpga_fusion_mode udp_bi \
          --candidate_dpga_active_adapters dpfm1,agf1 \
          --candidate_dpga_prior_embed_channels 24 \
          --candidate_dpga_adapter_residual_scale 0.2 \
          --candidate_dpga_adapter_bootstrap_scale 0.02 \
          --candidate_dpga_scale_multiplier 1.0 \
          --dpga_depth_cache_dir "$DEPTH" \
          --split_json "$SPLIT_JSON" \
          --split_name "$split" \
          --output_dir "$SEED_OUT/eval_regular_hard" \
          --tag "v19_seed${seed}_${lower}_${split}_vs_a0"
    done
  done
  if compgen -G "$SEED_OUT/eval_regular_hard/scout_eval_compare_v19_seed${seed}_*_val_regular_vs_a0.json" >/dev/null; then
    run_logged "v19_select_seed_${seed}" "$SEED_OUT/v19_seed${seed}_selection.log" \
      "$PY" experience_docx/tools/select_haze4k_multimetric_checkpoint.py \
        --eval_dir "$SEED_OUT/eval_regular_hard" \
        --candidate_prefix "v19_seed${seed}" \
        --checkpoint_labels model_5 model_10 model_15 model_20 Final \
        --output_json "$SEED_OUT/v19_seed${seed}_multimetric_checkpoint_selection.json" \
        --output_csv "$SEED_OUT/v19_seed${seed}_multimetric_checkpoint_selection.csv"
    [ -f "$SEED_OUT/v19_seed${seed}_multimetric_checkpoint_selection.json" ] && selection_jsons+=("$SEED_OUT/v19_seed${seed}_multimetric_checkpoint_selection.json")
  fi
done

for item in $HYGIENE; do
  IFS=: read -r name clip use_ema <<<"$item"
  OUT=$EVID/hygiene_$name
  args=(--udp_repo "$UDP_REPO" --data_dir "$DATA" --depth_cache_dir "$DEPTH" --a0_checkpoint "$A0" --official_checkpoint "$OFFICIAL_CKPT" --split_json "$SPLIT_JSON" --model_name "ConvIR-Haze4K-v1.9-Hygiene-${name}-20260606" --output_dir "$OUT" --seed 3407 --epochs 8 --batch_size 4 --grad_clip_norm "$clip")
  if [ "$use_ema" = "1" ]; then
    args+=(--ema)
  fi
  run_logged "v19_hygiene_${name}" "$OUT/train_hygiene_${name}.log" "$PY" experience_docx/tools/train_haze4k_v19_conditional_student.py "${args[@]}"
done

AGG_OUT=$EVID/v19_multiseed_aggregate
mkdir -p "$AGG_OUT"
if [ "${#selection_jsons[@]}" -gt 0 ]; then
  run_logged "v19_multiseed_aggregate" "$AGG_OUT/v19_multiseed_aggregate.log" \
    "$PY" experience_docx/tools/aggregate_haze4k_multiseed_checkpoint_metrics.py \
      --selection_jsons "${selection_jsons[@]}" \
      --output_json "$AGG_OUT/v19_multiseed_aggregate_summary.json" \
      --output_csv "$AGG_OUT/v19_multiseed_aggregate_metrics.csv"
else
  log_status "v19_multiseed_aggregate_skipped reason=no_selection_jsons"
fi

log_status "state=COMPLETED_QUEUE_PENDING_SYNC"
printf 'V19_CONDITIONAL_TEACHER_QUEUE_OK\n'
