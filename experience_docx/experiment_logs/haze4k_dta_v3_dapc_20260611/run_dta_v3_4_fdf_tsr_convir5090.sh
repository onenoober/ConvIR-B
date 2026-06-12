#!/usr/bin/env bash
set -euo pipefail

VARIANT=${VARIANT:-e2_tiny_residual}
STAGE=${STAGE:-quick5full}
SEED=${SEED:-3407}
BASE=${BASE:-/home/caozhiyang/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-4-fdf-tsr-finetune}
PY=${PY:-$BASE/envs/convir-cu128/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
MAX_IMAGES=${MAX_IMAGES:-0}
RUN_TEST=${RUN_TEST:-1}
RUN_TRAIN_CONTROLS=${RUN_TRAIN_CONTROLS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "$STAGE" in
  smoke) NUM_EPOCH=1; STOP_EPOCH=1; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  quick3full) NUM_EPOCH=3; STOP_EPOCH=3; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  quick5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac

SAFE_FLAG=()
SAFE_EVAL_FLAG=()
ROUTER_FLAG=()
ROUTER_EVAL_FLAG=()
case "$VARIANT" in
  e1_feature_only)
    TRAIN_SCOPE=dta_fdf_feature_only
    LR=0.000050
    FEATURE_STRENGTH=0.12
    FEATURE_BIAS=3.0
    DEPTH_SCALE=0.0
    MASK_EASY=0.0
    MASK_DENSE=0.0
    SAFE_CLIP=0.0
    SAFE_PHYS=0.0
    SAFE_LEARNED=0.0
    SAFE_GATE_BIAS=3.0
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.004
    CVAR_W=0.004
    GROUP_W=0.003
    PATCH_SSIM_W=0.003
    CF_W=0.000
    ;;
  e2_tiny_residual)
    TRAIN_SCOPE=dta_fdf_tsr_residual
    LR=0.000045
    FEATURE_STRENGTH=0.12
    FEATURE_BIAS=3.0
    DEPTH_SCALE=0.0
    MASK_EASY=0.0
    MASK_DENSE=0.0
    SAFE_CLIP=0.03
    SAFE_PHYS=0.0
    SAFE_LEARNED=1.0
    SAFE_GATE_BIAS=3.0
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.006
    CVAR_W=0.006
    GROUP_W=0.004
    PATCH_SSIM_W=0.004
    CF_W=0.000
    SAFE_FLAG=(--dta_safe_mix_enabled)
    SAFE_EVAL_FLAG=(--candidate_dta_safe_mix_enabled)
    ;;
  e3_tsr_full)
    TRAIN_SCOPE=dta_fdf_tsr_full
    LR=0.000040
    FEATURE_STRENGTH=0.12
    FEATURE_BIAS=3.0
    DEPTH_SCALE=0.0
    MASK_EASY=0.0
    MASK_DENSE=0.0
    SAFE_CLIP=0.03
    SAFE_PHYS=0.0
    SAFE_LEARNED=1.0
    SAFE_GATE_BIAS=3.0
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.006
    CVAR_W=0.006
    GROUP_W=0.004
    PATCH_SSIM_W=0.004
    CF_W=0.001
    SAFE_FLAG=(--dta_safe_mix_enabled)
    SAFE_EVAL_FLAG=(--candidate_dta_safe_mix_enabled)
    ROUTER_FLAG=(--dta_router_fusion_enabled --dta_router_image_gate_limit 1.0 --dta_router_patch_gate_limit 1.0 --dta_router_patch_size 32 --dta_router_image_bias 3.0 --dta_router_patch_bias 3.0)
    ROUTER_EVAL_FLAG=(--candidate_dta_router_fusion_enabled --candidate_dta_router_image_gate_limit 1.0 --candidate_dta_router_patch_gate_limit 1.0 --candidate_dta_router_patch_size 32 --candidate_dta_router_image_bias 3.0 --candidate_dta_router_patch_bias 3.0)
    ;;
  *) echo "Unsupported VARIANT=$VARIANT" >&2; exit 64 ;;
esac

RUN_ID=v34_fdf_tsr_${VARIANT}_seed${SEED}_${STAGE}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-4-FDF-TSR-${VARIANT}-seed${SEED}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_4_${RUN_ID}_train.log
mkdir -p "$EVID"
{
  echo "dta_v3_4_fdf_tsr_start run_id=$RUN_ID variant=$VARIANT stage=$STAGE $(date --iso-8601=seconds)"
  echo "state=USER_EXPLICIT_TEST_OVERRIDE_ONE_SHOT"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "train_scope=$TRAIN_SCOPE lr=$LR feature_strength=$FEATURE_STRENGTH feature_bias=$FEATURE_BIAS safe_clip=$SAFE_CLIP safe_phys=$SAFE_PHYS safe_learned=$SAFE_LEARNED safe_gate_bias=$SAFE_GATE_BIAS"
  echo "fine_tune_from_official_a0=true"
  echo "gate_policy=widest_user_override"
  echo "locked_test_touched=$RUN_TEST"
  echo "locked_test_policy_note=single explicit user override; no checkpoint/gate selection from test"
} | tee -a "$STATUS"

for p in "$PY" "$DATA" "$A0" "$DEPTH"; do
  if [[ ! -e "$p" ]]; then
    echo "DTA_V3_4_FDF_TSR_MISSING_PATH $p" | tee -a "$STATUS"
    exit 3
  fi
done

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
cd "$WORK/Dehazing/ITS"
if [[ "${FORCE:-0}" != "1" && -f "results/$MODEL_NAME/Training-Results/Final.pkl" ]]; then
  echo "dta_v3_4_fdf_tsr_train_skip_existing model=$MODEL_NAME $(date --iso-8601=seconds)" | tee -a "$STATUS"
else
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
    --dta_gate_bias 3.0 \
    --dta_gate_limit 1.0 \
    --dta_gamma_limit 0.50 \
    --dta_beta_limit 0.50 \
    --dta_confidence_floor 0.30 \
    --dta_r0_residual_scale 0.0 \
    --dta_depth_residual_scale "$DEPTH_SCALE" \
    --dta_depth_mask_easy_budget "$MASK_EASY" \
    --dta_depth_mask_dense_budget "$MASK_DENSE" \
    --dta_depth_mask_density_thresh 0.35 \
    --dta_depth_mask_bias -4.0 \
    --dta_phys_t_min 0.10 \
    "${SAFE_FLAG[@]}" \
    --dta_safe_mix_delta_clip "$SAFE_CLIP" \
    --dta_safe_mix_phys_weight "$SAFE_PHYS" \
    --dta_safe_mix_learned_weight "$SAFE_LEARNED" \
    --dta_safe_mix_gate_limit 1.0 \
    --dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
    "${ROUTER_FLAG[@]}" \
    --dta_feature_fusion_enabled \
    --dta_feature_fusion_strength "$FEATURE_STRENGTH" \
    --dta_feature_fusion_gate_limit 1.0 \
    --dta_feature_fusion_gate_bias "$FEATURE_BIAS" \
    --dta_use_trans_gt \
    --dta_rank_weight 0.0 \
    --dta_tv_weight 0.0 \
    --dta_proxy_weight 0.0 \
    --dta_trans_weight 0.0 \
    --dta_trans_log_weight "$TRANS_LOG_W" \
    --dta_trans_nll_weight "$TRANS_NLL_W" \
    --dta_phys_weight 0.0 \
    --dta_preserve_weight 0.006 \
    --dta_preserve_trans_thresh 0.80 \
    --dta_reference_checkpoint "$A0" \
    --dta_ref_preserve_weight "$REF_W" \
    --dta_tail_guard_weight 0.0 \
    --dta_mask_budget_weight 0.0 \
    --dta_light_tail_hinge_weight 0.0 \
    --dta_light_ssim_hinge_weight 0.0 \
    --dta_cvar_tail_weight "$CVAR_W" \
    --dta_cvar_tail_margin 0.0 \
    --dta_cvar_tail_topk 0.10 \
    --dta_group_tail_weight "$GROUP_W" \
    --dta_patch_ssim_cvar_weight "$PATCH_SSIM_W" \
    --dta_patch_ssim_cvar_margin 0.0 \
    --dta_patch_ssim_cvar_topk 0.10 \
    --dta_counterfactual_gate_weight "$CF_W" \
    --dta_counterfactual_modes zero,normal \
    2>&1 | tee "$TRAIN_LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_4_fdf_tsr_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
fi

CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_4_FDF_TSR_MISSING_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

run_matrix() {
  local root_split=$1
  local depth_split=$2
  local matrix_suffix=$3
  local split_args=()
  local manifest=$EVID/dta_v3_4_${RUN_ID}_${matrix_suffix}_matrix_manifest.json
  printf '{"runs":[\n' > "$manifest"
  local first=1
  for EVAL_MODE in invert zero shuffle normal; do
    local EVAL_RUN=${RUN_ID}_${matrix_suffix}_${EVAL_MODE}
    local COMPARE_DIR=$EVID/dta_v3_4_${EVAL_RUN}_compare
    mkdir -p "$COMPARE_DIR"
    set +e
    PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
      --data_dir "$DATA" \
      --original_checkpoint "$A0" \
      --original_arch official_convir \
      --original_name A0 \
      --candidate_checkpoint "$CANDIDATE" \
      --candidate_arch dta_v3 \
      --candidate_name "DTA_v34_${EVAL_RUN}" \
      --dta_depth_cache_dir "$DEPTH" \
      --dta_eval_depth_split "$depth_split" \
      --eval_root_split "$root_split" \
      --candidate_dta_variant v3 \
      --candidate_dta_depth_mode "$EVAL_MODE" \
      --candidate_dta_airlight_mode fallback \
      --candidate_dta_phase depth \
      --candidate_dta_ablation full \
      --candidate_dta_prior_channels 32 \
      --candidate_dta_gate_bias 3.0 \
      --candidate_dta_gate_limit 1.0 \
      --candidate_dta_gamma_limit 0.50 \
      --candidate_dta_beta_limit 0.50 \
      --candidate_dta_confidence_floor 0.30 \
      --candidate_dta_r0_residual_scale 0.0 \
      --candidate_dta_depth_residual_scale "$DEPTH_SCALE" \
      --candidate_dta_depth_mask_easy_budget "$MASK_EASY" \
      --candidate_dta_depth_mask_dense_budget "$MASK_DENSE" \
      --candidate_dta_depth_mask_density_thresh 0.35 \
      --candidate_dta_depth_mask_bias -4.0 \
      --candidate_dta_phys_t_min 0.10 \
      "${SAFE_EVAL_FLAG[@]}" \
      --candidate_dta_safe_mix_delta_clip "$SAFE_CLIP" \
      --candidate_dta_safe_mix_phys_weight "$SAFE_PHYS" \
      --candidate_dta_safe_mix_learned_weight "$SAFE_LEARNED" \
      --candidate_dta_safe_mix_gate_limit 1.0 \
      --candidate_dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
      "${ROUTER_EVAL_FLAG[@]}" \
      --candidate_dta_feature_fusion_enabled \
      --candidate_dta_feature_fusion_strength "$FEATURE_STRENGTH" \
      --candidate_dta_feature_fusion_gate_limit 1.0 \
      --candidate_dta_feature_fusion_gate_bias "$FEATURE_BIAS" \
      --depth_shuffle_offset 137 \
      "${split_args[@]}" \
      --output_dir "$COMPARE_DIR" \
      --tag "$EVAL_RUN" \
      --max_images "$MAX_IMAGES" \
      2>&1 | tee "$EVID/dta_v3_4_${EVAL_RUN}_eval.log"
    local eval_rc=${PIPESTATUS[0]}
    set -e
    echo "dta_v3_4_fdf_tsr_eval_done rc=$eval_rc run_id=$EVAL_RUN $(date --iso-8601=seconds)" | tee -a "$STATUS"
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
    2>&1 | tee "$EVID/dta_v3_4_${RUN_ID}_${matrix_suffix}_aggregate.log"
  local agg_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_4_fdf_tsr_aggregate_done rc=$agg_rc run_id=$RUN_ID matrix=$matrix_suffix $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$agg_rc" -ne 0 ]]; then exit "$agg_rc"; fi
}

cd "$WORK"
if [[ "$RUN_TRAIN_CONTROLS" == "1" ]]; then
  run_matrix train train fallback_train
fi
if [[ "$RUN_TEST" == "1" ]]; then
  run_matrix test test fallback_test
fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/summarize_haze4k_dta_v34_fdf_tsr.py \
  --evidence_dir "$EVID" \
  --output_json "$EVID/dta_v3_4_fdf_tsr_summary.json" \
  --output_csv "$EVID/dta_v3_4_fdf_tsr_summary.csv" \
  2>&1 | tee "$EVID/dta_v3_4_${RUN_ID}_summary.log"
summary_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_4_fdf_tsr_summary_done rc=$summary_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$summary_rc" -ne 0 ]]; then exit "$summary_rc"; fi

if [[ "$RUN_TEST" == "1" ]]; then
  CONTACT_DIR=$EVID/tail_regression_contact_sheet/${RUN_ID}_fallback_test
  mkdir -p "$CONTACT_DIR"
  true_csv=$(ls "$EVID"/dta_v3_4_${RUN_ID}_fallback_test_invert_compare/scout_eval_per_image_*.csv | head -n 1)
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_contact_sheet.py \
    --data_dir "$DATA" --depth_cache_dir "$DEPTH" --depth_split test --root_split test \
    --per_image_csv "$true_csv" \
    --a0_checkpoint "$A0" --candidate_checkpoint "$CANDIDATE" --candidate_arch dta_v3 \
    --output_dir "$CONTACT_DIR" --tag "$RUN_ID" --count 16 \
    --dta_variant v3 --dta_depth_mode invert --dta_airlight_mode fallback --dta_phase depth --dta_ablation full \
    --dta_prior_channels 32 --dta_gate_bias 3.0 --dta_gate_limit 1.0 --dta_gamma_limit 0.50 --dta_beta_limit 0.50 \
    --dta_confidence_floor 0.30 --dta_r0_residual_scale 0.0 --dta_depth_residual_scale "$DEPTH_SCALE" \
    --dta_depth_mask_easy_budget "$MASK_EASY" --dta_depth_mask_dense_budget "$MASK_DENSE" --dta_depth_mask_bias -4.0 \
    "${SAFE_FLAG[@]}" --dta_safe_mix_delta_clip "$SAFE_CLIP" --dta_safe_mix_phys_weight "$SAFE_PHYS" \
    --dta_safe_mix_learned_weight "$SAFE_LEARNED" --dta_safe_mix_gate_limit 1.0 --dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
    "${ROUTER_FLAG[@]}" \
    --dta_feature_fusion_enabled --dta_feature_fusion_strength "$FEATURE_STRENGTH" --dta_feature_fusion_gate_limit 1.0 --dta_feature_fusion_gate_bias "$FEATURE_BIAS" \
    2>&1 | tee "$EVID/dta_v3_4_${RUN_ID}_contact_sheet.log"
  contact_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_4_fdf_tsr_contact_sheet_done rc=$contact_rc run_id=$RUN_ID output=$CONTACT_DIR $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$contact_rc" -ne 0 ]]; then exit "$contact_rc"; fi
fi

echo "DTA_V3_4_FDF_TSR_OK run_id=$RUN_ID checkpoint=$CANDIDATE" | tee -a "$STATUS"
