#!/usr/bin/env bash
set -euo pipefail

VARIANT=${VARIANT:-u1_tau_l1_s004_g025_a006}
STAGE=${STAGE:-quick5full}
SEED=${SEED:-3407}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-7-u-tqs-mix-phased1}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613
STATUS=$EVID/status_phase_d1_tau_training.txt
MAX_IMAGES=${MAX_IMAGES:-0}
RUN_TRAIN_CONTROLS=${RUN_TRAIN_CONTROLS:-1}
USE_SPLIT=${USE_SPLIT:-1}
FOLD=${FOLD:-0}
STAGE_SCREEN_ONLY=${DTA_V37_STAGE_SCREEN_ONLY:-1}
STAGE_SCREEN_VARIANTS=${DTA_V37_STAGE_SCREEN_VARIANTS:-u1_tau_l1_s004_g025_a006,u2_tau_l3_s004_g015_a006,u3_tau_l2_s002_g025_a006}
STAGE_SCREEN_FOLDS=${DTA_V37_STAGE_SCREEN_FOLDS:-0,1}
STAGE_SCREEN_SEEDS=${DTA_V37_STAGE_SCREEN_SEEDS:-3407,3411}
RUN_BATCH_TAG=${DTA_V37_RUN_BATCH_TAG:-}
SPLIT_JSON=${SPLIT_JSON:-$WORK/experience_docx/experiment_logs/haze4k_dta_v3_6_hrcs_20260613/dta_v3_6_haze4k_oof_splits_seed3407.json}
TRAIN_SPLIT=${TRAIN_SPLIT:-fold${FOLD}_train}
EVAL_SPLIT=${EVAL_SPLIT:-fold${FOLD}_val}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

csv_has_value() {
  local csv=",$1,"
  local value="$2"
  [[ "$csv" == *",$value,"* ]]
}

case "$STAGE" in
  smoke) NUM_EPOCH=1; STOP_EPOCH=1; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  quick5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  formal20) NUM_EPOCH=20; STOP_EPOCH=20; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=5 ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac

case "$VARIANT" in
  u1_tau_l1_s004_g025_a006)
    TRAIN_SCOPE=dta_fdf_feature_only
    LR=0.000050
    FEATURE_STRENGTH=0.04
    FEATURE_GATE_LIMIT=0.25
    FEATURE_BIAS=-2.0
    AIR_W=0.006
    AIR_NLL_W=0.001
    TRANS_LOG_W=0.015
    TRANS_NLL_W=0.004
    REF_W=0.006
    REF_MSE_W=0.006
    CVAR_W=0.006
    GROUP_W=0.006
    PATCH_SSIM_W=0.004
    LIGHT_W=0.004
    ;;
  u2_tau_l3_s004_g015_a006)
    TRAIN_SCOPE=dta_fdf_feature_only
    LR=0.000050
    FEATURE_STRENGTH=0.04
    FEATURE_GATE_LIMIT=0.15
    FEATURE_BIAS=-2.0
    AIR_W=0.006
    AIR_NLL_W=0.001
    TRANS_LOG_W=0.015
    TRANS_NLL_W=0.004
    REF_W=0.006
    REF_MSE_W=0.006
    CVAR_W=0.006
    GROUP_W=0.006
    PATCH_SSIM_W=0.004
    LIGHT_W=0.004
    ;;
  u3_tau_l2_s002_g025_a006)
    TRAIN_SCOPE=dta_fdf_feature_only
    LR=0.000050
    FEATURE_STRENGTH=0.02
    FEATURE_GATE_LIMIT=0.25
    FEATURE_BIAS=-2.0
    AIR_W=0.006
    AIR_NLL_W=0.001
    TRANS_LOG_W=0.015
    TRANS_NLL_W=0.004
    REF_W=0.006
    REF_MSE_W=0.006
    CVAR_W=0.006
    GROUP_W=0.006
    PATCH_SSIM_W=0.004
    LIGHT_W=0.004
    ;;
  *) echo "Unsupported VARIANT=$VARIANT" >&2; exit 64 ;;
esac

SPLIT_TAG=""
if [[ "$USE_SPLIT" == "1" ]]; then
  SPLIT_TAG="_f${FOLD}"
fi
RUN_ID=v35_fdf_rcs_${VARIANT}_seed${SEED}${SPLIT_TAG}_${STAGE}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-7-TAU-${VARIANT}-seed${SEED}${SPLIT_TAG}-${STAGE}
if [[ -n "$RUN_BATCH_TAG" ]]; then
  RUN_ID=${RUN_ID}_${RUN_BATCH_TAG}
  MODEL_NAME=${MODEL_NAME}-${RUN_BATCH_TAG}
fi
TRAIN_LOG=$EVID/dta_v3_7_${RUN_ID}_train.log
mkdir -p "$EVID"

if [[ "$STAGE_SCREEN_ONLY" == "1" ]]; then
  if ! csv_has_value "$STAGE_SCREEN_VARIANTS" "$VARIANT" || \
     ! csv_has_value "$STAGE_SCREEN_FOLDS" "$FOLD" || \
     ! csv_has_value "$STAGE_SCREEN_SEEDS" "$SEED"; then
    {
      echo "DTA_V3_7_TAU_CANDIDATE_SKIPPED run_id=$RUN_ID reason=stage_screen_only variant=$VARIANT fold=$FOLD seed=$SEED stage=$STAGE $(date --iso-8601=seconds)"
      echo "stage_screen_policy=variants:$STAGE_SCREEN_VARIANTS folds:$STAGE_SCREEN_FOLDS seeds:$STAGE_SCREEN_SEEDS"
      echo "set DTA_V37_STAGE_SCREEN_ONLY=0 only after a documented screen-to-formal promotion decision"
    } | tee -a "$STATUS"
    exit 0
  fi
fi

{
  echo "dta_v3_7_tau_candidate_start run_id=$RUN_ID variant=$VARIANT stage=$STAGE $(date --iso-8601=seconds)"
  echo "state=RUNNING_TRAIN_DERIVED_INTEGRATED_TAU"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "train_scope=$TRAIN_SCOPE lr=$LR feature_strength=$FEATURE_STRENGTH feature_gate_limit=$FEATURE_GATE_LIMIT feature_bias=$FEATURE_BIAS"
  echo "loss_weights trans_log=$TRANS_LOG_W trans_nll=$TRANS_NLL_W air=$AIR_W air_nll=$AIR_NLL_W ref=$REF_W ref_mse=$REF_MSE_W cvar=$CVAR_W group=$GROUP_W patch_ssim=$PATCH_SSIM_W light=$LIGHT_W"
  echo "partial_load_rule=init official A0 with --init_model_partial --partial_new_prefixes DTA.; all DTA modules including airlight heads are newly initialized"
  echo "locked_test_touched=false"
  echo "use_split=$USE_SPLIT fold=$FOLD split_json=$SPLIT_JSON train_split=$TRAIN_SPLIT eval_split=$EVAL_SPLIT"
} | tee -a "$STATUS"

for p in "$PY" "$DATA" "$A0" "$DEPTH" "$SPLIT_JSON"; do
  if [[ ! -e "$p" ]]; then
    echo "DTA_V3_7_TAU_CANDIDATE_MISSING_PATH $p" | tee -a "$STATUS"
    exit 3
  fi
done

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"

cd "$WORK/Dehazing/ITS"
if [[ "${FORCE:-0}" != "1" && -f "results/$MODEL_NAME/Training-Results/Final.pkl" ]]; then
  echo "dta_v3_7_tau_train_skip_existing model=$MODEL_NAME $(date --iso-8601=seconds)" | tee -a "$STATUS"
  CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
else
  TRAIN_SPLIT_ARGS=()
  if [[ "$USE_SPLIT" == "1" ]]; then
    TRAIN_SPLIT_ARGS=(--split_json "$SPLIT_JSON" --split_name "$TRAIN_SPLIT")
  fi
  set +e
  PYTHONUNBUFFERED=1 "$PY" main.py \
    --model_name "$MODEL_NAME" \
    --data Haze4K \
    --version base \
    --fam_mode original \
    --arch dta_v3 \
    --dta_variant v3 \
    --seed "$SEED" \
    --mode train \
    --data_dir "$DATA" \
    --batch_size 4 \
    --learning_rate "$LR" \
    --weight_decay 0.0001 \
    --num_epoch "$NUM_EPOCH" \
    --stop_epoch "$STOP_EPOCH" \
    --print_freq 50 \
    --num_worker 4 \
    --save_freq "$SAVE_FREQ" \
    --valid_freq "$VALID_FREQ" \
    --valid_root_split train \
    --mod_stats_freq "$MOD_STATS_FREQ" \
    --mod_stats_batches 16 \
    --grad_clip_norm 0.0005 \
    --dta_grad_clip_norm 0.030 \
    --init_model "$A0" \
    --init_model_partial \
    --partial_new_prefixes DTA. \
    --train_scope "$TRAIN_SCOPE" \
    --dta_depth_cache_dir "$DEPTH" \
    --dta_train_depth_split train \
    --dta_eval_depth_split train \
    --dta_require_depth \
    --dta_depth_mode invert \
    --dta_phase depth \
    --dta_ablation full \
    --dta_prior_channels 32 \
    --dta_gate_bias -2.0 \
    --dta_gate_limit 0.25 \
    --dta_gamma_limit 0.20 \
    --dta_beta_limit 0.10 \
    --dta_confidence_floor 0.30 \
    --dta_r0_residual_scale 0.0 \
    --dta_depth_residual_scale 0.0 \
    --dta_depth_mask_easy_budget 0.0 \
    --dta_depth_mask_dense_budget 0.0 \
    --dta_depth_mask_density_thresh 0.35 \
    --dta_depth_mask_bias -4.0 \
    --dta_phys_t_min 0.10 \
    --dta_feature_fusion_enabled \
    --dta_feature_fusion_strength "$FEATURE_STRENGTH" \
    --dta_feature_fusion_gate_limit "$FEATURE_GATE_LIMIT" \
    --dta_feature_fusion_gate_bias "$FEATURE_BIAS" \
    --dta_use_trans_gt \
    --dta_rank_weight 0.0 \
    --dta_tv_weight 0.0 \
    --dta_proxy_weight 0.0 \
    --dta_trans_weight 0.0 \
    --dta_trans_log_weight "$TRANS_LOG_W" \
    --dta_trans_nll_weight "$TRANS_NLL_W" \
    --dta_airlight_weight "$AIR_W" \
    --dta_airlight_nll_weight "$AIR_NLL_W" \
    --dta_phys_weight 0.0 \
    --dta_preserve_weight 0.006 \
    --dta_preserve_trans_thresh 0.80 \
    --dta_reference_checkpoint "$A0" \
    --dta_ref_preserve_weight "$REF_W" \
    --dta_ref_mse_regression_weight "$REF_MSE_W" \
    --dta_ref_mse_regression_margin 0.0 \
    --dta_tail_guard_weight 0.0 \
    --dta_mask_budget_weight 0.0 \
    --dta_feature_gate_budget_weight 0.002 \
    --dta_feature_action_budget_weight 0.004 \
    --dta_light_tail_hinge_weight "$LIGHT_W" \
    --dta_light_tail_hinge_margin 0.0 \
    --dta_light_ssim_hinge_weight 0.0 \
    --dta_cvar_tail_weight "$CVAR_W" \
    --dta_cvar_tail_margin 0.0 \
    --dta_cvar_tail_topk 0.10 \
    --dta_group_tail_weight "$GROUP_W" \
    --dta_patch_ssim_cvar_weight "$PATCH_SSIM_W" \
    --dta_patch_ssim_cvar_margin 0.0 \
    --dta_patch_ssim_cvar_topk 0.10 \
    --dta_counterfactual_gate_weight 0.0 \
    "${TRAIN_SPLIT_ARGS[@]}" \
    2>&1 | tee "$TRAIN_LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_7_tau_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
  CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
fi

if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_7_TAU_CANDIDATE_MISSING_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

run_matrix() {
  local root_split=$1
  local depth_split=$2
  local matrix_suffix=$3
  local split_args=()
  if [[ "$USE_SPLIT" == "1" ]]; then
    split_args=(--split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT")
  fi
  local manifest=$EVID/dta_v3_7_${RUN_ID}_${matrix_suffix}_matrix_manifest.json
  printf '{"runs":[\n' > "$manifest"
  local first=1
  for EVAL_MODE in invert zero shuffle normal; do
    local EVAL_RUN=${RUN_ID}_${matrix_suffix}_${EVAL_MODE}
    local COMPARE_DIR=$EVID/dta_v3_7_${EVAL_RUN}_compare
    mkdir -p "$COMPARE_DIR"
    set +e
    PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
      --data_dir "$DATA" \
      --original_checkpoint "$A0" \
      --original_arch official_convir \
      --original_name A0 \
      --candidate_checkpoint "$CANDIDATE" \
      --candidate_arch dta_v3 \
      --candidate_name "DTA_v37_${EVAL_RUN}" \
      --dta_depth_cache_dir "$DEPTH" \
      --dta_eval_depth_split "$depth_split" \
      --eval_root_split "$root_split" \
      --candidate_dta_variant v3 \
      --candidate_dta_depth_mode "$EVAL_MODE" \
      --candidate_dta_airlight_mode fallback \
      --candidate_dta_phase depth \
      --candidate_dta_ablation full \
      --candidate_dta_prior_channels 32 \
      --candidate_dta_gate_bias -2.0 \
      --candidate_dta_gate_limit 0.25 \
      --candidate_dta_gamma_limit 0.20 \
      --candidate_dta_beta_limit 0.10 \
      --candidate_dta_confidence_floor 0.30 \
      --candidate_dta_r0_residual_scale 0.0 \
      --candidate_dta_depth_residual_scale 0.0 \
      --candidate_dta_depth_mask_easy_budget 0.0 \
      --candidate_dta_depth_mask_dense_budget 0.0 \
      --candidate_dta_depth_mask_density_thresh 0.35 \
      --candidate_dta_depth_mask_bias -4.0 \
      --candidate_dta_phys_t_min 0.10 \
      --candidate_dta_feature_fusion_enabled \
      --candidate_dta_feature_fusion_strength "$FEATURE_STRENGTH" \
      --candidate_dta_feature_fusion_gate_limit "$FEATURE_GATE_LIMIT" \
      --candidate_dta_feature_fusion_gate_bias "$FEATURE_BIAS" \
      --depth_shuffle_offset 137 \
      --include_trans_stats \
      "${split_args[@]}" \
      --output_dir "$COMPARE_DIR" \
      --tag "$EVAL_RUN" \
      --max_images "$MAX_IMAGES" \
      2>&1 | tee "$EVID/dta_v3_7_${EVAL_RUN}_eval.log"
    local eval_rc=${PIPESTATUS[0]}
    set -e
    echo "dta_v3_7_tau_eval_done rc=$eval_rc run_id=$EVAL_RUN $(date --iso-8601=seconds)" | tee -a "$STATUS"
    if [[ "$eval_rc" -ne 0 ]]; then exit "$eval_rc"; fi
    if [[ "$first" -eq 0 ]]; then printf ',\n' >> "$manifest"; fi
    first=0
    local label="$EVAL_MODE"
    if [[ "$EVAL_MODE" == "invert" ]]; then label="true"; fi
    printf '  {"label":"%s","train_depth":"invert","eval_depth":"%s","airlight_mode":"fallback","root_split":"%s","compare_dir":"%s"}' "$label" "$EVAL_MODE" "$root_split" "$COMPARE_DIR" >> "$manifest"
  done
  printf '\n]}\n' >> "$manifest"
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/aggregate_haze4k_dta_v3_controls.py \
    --manifest "$manifest" \
    --output_matrix_json "$EVID/train_eval_depth_matrix_${RUN_ID}_${matrix_suffix}.json" \
    --output_matrix_csv "$EVID/train_eval_depth_matrix_${RUN_ID}_${matrix_suffix}.csv" \
    --output_attribution_csv "$EVID/r0_vs_rdepth_attribution_${RUN_ID}_${matrix_suffix}.csv" \
    --baseline_label zero \
    --true_label true \
    2>&1 | tee "$EVID/dta_v3_7_${RUN_ID}_${matrix_suffix}_aggregate.log"
  local agg_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_7_tau_aggregate_done rc=$agg_rc run_id=$RUN_ID matrix=$matrix_suffix $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$agg_rc" -ne 0 ]]; then exit "$agg_rc"; fi
}

cd "$WORK"
if [[ "$RUN_TRAIN_CONTROLS" == "1" ]]; then
  run_matrix train train fallback_train
fi

echo "DTA_V3_7_TAU_CANDIDATE_OK run_id=$RUN_ID checkpoint=$CANDIDATE" | tee -a "$STATUS"
