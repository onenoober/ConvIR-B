#!/usr/bin/env bash
set -euo pipefail

VARIANT=${VARIANT:-d3_router}
STAGE=${STAGE:-triage5full}
SEED=${SEED:-3407}
FOLD=${FOLD:-0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v33-routerfusion}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
INIT_CKPT=${INIT_CKPT:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v32-safemix/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-2-SafeMix-c3_full-seed3407-f0-scout5full/Training-Results/Final.pkl}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
TRAIN_SPLIT=${TRAIN_SPLIT:-fold${FOLD}_train}
EVAL_SPLIT=${EVAL_SPLIT:-fold${FOLD}_val}
MAX_IMAGES=${MAX_IMAGES:-0}
RUN_DIAG=${RUN_DIAG:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "$STAGE" in
  smoke) NUM_EPOCH=1; STOP_EPOCH=1; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  triage5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac

ROUTER_FLAG=()
ROUTER_EVAL_FLAG=()
case "$VARIANT" in
  d1_loss)
    TRAIN_SCOPE=dta_safemix_full
    SAFE_PHYS=0.25
    SAFE_LEARNED=0.75
    SAFE_CLIP=0.06
    SAFE_GATE_BIAS=-2.5
    LR=0.000035
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.020
    LIGHT_TAIL_W=0.010
    LIGHT_SSIM_W=0.005
    CVAR_W=0.020
    GROUP_W=0.010
    PATCH_SSIM_W=0.010
    CF_W=0.000
    ;;
  d2_lowphys)
    TRAIN_SCOPE=dta_safemix_full
    SAFE_PHYS=0.15
    SAFE_LEARNED=0.85
    SAFE_CLIP=0.05
    SAFE_GATE_BIAS=-2.3
    LR=0.000035
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.020
    LIGHT_TAIL_W=0.010
    LIGHT_SSIM_W=0.005
    CVAR_W=0.020
    GROUP_W=0.010
    PATCH_SSIM_W=0.012
    CF_W=0.000
    ;;
  d3_router)
    TRAIN_SCOPE=dta_routerfusion_full
    SAFE_PHYS=0.15
    SAFE_LEARNED=0.85
    SAFE_CLIP=0.05
    SAFE_GATE_BIAS=-2.2
    LR=0.000035
    TRANS_LOG_W=0.010
    TRANS_NLL_W=0.002
    REF_W=0.015
    LIGHT_TAIL_W=0.008
    LIGHT_SSIM_W=0.004
    CVAR_W=0.018
    GROUP_W=0.012
    PATCH_SSIM_W=0.015
    CF_W=0.001
    ROUTER_FLAG=(--dta_router_fusion_enabled --dta_router_image_gate_limit 1.0 --dta_router_patch_gate_limit 1.0 --dta_router_patch_size 32 --dta_router_image_bias 2.2 --dta_router_patch_bias 2.0)
    ROUTER_EVAL_FLAG=(--candidate_dta_router_fusion_enabled --candidate_dta_router_image_gate_limit 1.0 --candidate_dta_router_patch_gate_limit 1.0 --candidate_dta_router_patch_size 32 --candidate_dta_router_image_bias 2.2 --candidate_dta_router_patch_bias 2.0)
    ;;
  *) echo "Unsupported VARIANT=$VARIANT" >&2; exit 64 ;;
esac

RUN_ID=v33_routerfusion_${VARIANT}_seed${SEED}_f${FOLD}_${STAGE}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-3-RouterFusion-${VARIANT}-seed${SEED}-f${FOLD}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_3_${RUN_ID}_train.log
mkdir -p "$EVID"
{
  echo "dta_v3_3_routerfusion_train_start run_id=$RUN_ID variant=$VARIANT stage=$STAGE $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "init_ckpt=$INIT_CKPT"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "train_split=$TRAIN_SPLIT"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "train_scope=$TRAIN_SCOPE safe_phys=$SAFE_PHYS safe_learned=$SAFE_LEARNED safe_clip=$SAFE_CLIP safe_gate_bias=$SAFE_GATE_BIAS cvar=$CVAR_W group=$GROUP_W patch_ssim=$PATCH_SSIM_W cf=$CF_W"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "DTA_V3_3_ROUTERFUSION_MISSING_INIT $INIT_CKPT" | tee -a "$STATUS"
  exit 3
fi

cd "$WORK/Dehazing/ITS"
if [[ "${FORCE:-0}" != "1" && -f "results/$MODEL_NAME/Training-Results/Final.pkl" ]]; then
  echo "dta_v3_3_routerfusion_train_skip_existing model=$MODEL_NAME $(date --iso-8601=seconds)" | tee -a "$STATUS"
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
    --init_model "$INIT_CKPT" \
    --init_model_partial \
    --partial_new_prefixes DTA.router_image_head.,DTA.router_patch_head. \
    --train_scope "$TRAIN_SCOPE" \
    --dta_depth_cache_dir "$DEPTH" \
    --dta_train_depth_split train \
    --dta_eval_depth_split train \
    --dta_require_depth \
    --dta_depth_mode invert \
    --dta_phase depth \
    --dta_ablation full \
    --dta_prior_channels 32 \
    --dta_gate_bias -5.0 \
    --dta_gate_limit 0.18 \
    --dta_gamma_limit 0.28 \
    --dta_beta_limit 0.14 \
    --dta_confidence_floor 0.30 \
    --dta_r0_residual_scale 0.0 \
    --dta_depth_residual_scale 0.08 \
    --dta_depth_mask_easy_budget 0.04 \
    --dta_depth_mask_dense_budget 0.14 \
    --dta_depth_mask_density_thresh 0.35 \
    --dta_depth_mask_bias -4.0 \
    --dta_phys_t_min 0.10 \
    --dta_safe_mix_enabled \
    --dta_safe_mix_delta_clip "$SAFE_CLIP" \
    --dta_safe_mix_phys_weight "$SAFE_PHYS" \
    --dta_safe_mix_learned_weight "$SAFE_LEARNED" \
    --dta_safe_mix_gate_limit 1.0 \
    --dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
    "${ROUTER_FLAG[@]}" \
    --dta_use_trans_gt \
    --dta_rank_weight 0.0 \
    --dta_tv_weight 0.0 \
    --dta_proxy_weight 0.0 \
    --dta_trans_weight 0.0 \
    --dta_trans_log_weight "$TRANS_LOG_W" \
    --dta_trans_nll_weight "$TRANS_NLL_W" \
    --dta_phys_weight 0.002 \
    --dta_preserve_weight 0.015 \
    --dta_preserve_trans_thresh 0.80 \
    --dta_reference_checkpoint "$A0" \
    --dta_ref_preserve_weight "$REF_W" \
    --dta_tail_guard_weight 0.0 \
    --dta_tail_guard_margin 0.0 \
    --dta_mask_budget_weight 0.001 \
    --dta_light_tail_hinge_weight "$LIGHT_TAIL_W" \
    --dta_light_tail_hinge_margin 0.0 \
    --dta_light_tail_hinge_topk 0.10 \
    --dta_light_hinge_bright_thresh 0.82 \
    --dta_light_hinge_texture_thresh 0.004 \
    --dta_light_ssim_hinge_weight "$LIGHT_SSIM_W" \
    --dta_light_ssim_hinge_margin 0.0 \
    --dta_cvar_tail_weight "$CVAR_W" \
    --dta_cvar_tail_margin 0.0 \
    --dta_cvar_tail_topk 0.10 \
    --dta_group_tail_weight "$GROUP_W" \
    --dta_patch_ssim_cvar_weight "$PATCH_SSIM_W" \
    --dta_patch_ssim_cvar_margin 0.0 \
    --dta_patch_ssim_cvar_topk 0.10 \
    --dta_counterfactual_gate_weight "$CF_W" \
    --dta_counterfactual_modes zero,normal \
    --split_json "$SPLIT_JSON" \
    --split_name "$TRAIN_SPLIT" \
    2>&1 | tee "$TRAIN_LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_3_routerfusion_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
fi

CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_3_ROUTERFUSION_MISSING_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

cd "$WORK"
for A_MODE in fallback gt; do
  MANIFEST=$EVID/dta_v3_3_${RUN_ID}_${A_MODE}_matrix_manifest.json
  printf '{"runs":[\n' > "$MANIFEST"
  first=1
  for EVAL_MODE in invert zero shuffle normal; do
    EVAL_RUN=${RUN_ID}_${A_MODE}_${EVAL_MODE}
    COMPARE_DIR=$EVID/dta_v3_3_${EVAL_RUN}_compare
    mkdir -p "$COMPARE_DIR"
    set +e
    PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
      --data_dir "$DATA" \
      --original_checkpoint "$A0" \
      --original_arch official_convir \
      --original_name A0 \
      --candidate_checkpoint "$CANDIDATE" \
      --candidate_arch dta_v3 \
      --candidate_name "DTA_v33_${EVAL_RUN}" \
      --dta_depth_cache_dir "$DEPTH" \
      --dta_eval_depth_split train \
      --candidate_dta_variant v3 \
      --candidate_dta_depth_mode "$EVAL_MODE" \
      --candidate_dta_airlight_mode "$A_MODE" \
      --candidate_dta_phase depth \
      --candidate_dta_ablation full \
      --candidate_dta_prior_channels 32 \
      --candidate_dta_gate_bias -5.0 \
      --candidate_dta_gate_limit 0.18 \
      --candidate_dta_gamma_limit 0.28 \
      --candidate_dta_beta_limit 0.14 \
      --candidate_dta_confidence_floor 0.30 \
      --candidate_dta_r0_residual_scale 0.0 \
      --candidate_dta_depth_residual_scale 0.08 \
      --candidate_dta_depth_mask_easy_budget 0.04 \
      --candidate_dta_depth_mask_dense_budget 0.14 \
      --candidate_dta_depth_mask_density_thresh 0.35 \
      --candidate_dta_depth_mask_bias -4.0 \
      --candidate_dta_phys_t_min 0.10 \
      --candidate_dta_safe_mix_enabled \
      --candidate_dta_safe_mix_delta_clip "$SAFE_CLIP" \
      --candidate_dta_safe_mix_phys_weight "$SAFE_PHYS" \
      --candidate_dta_safe_mix_learned_weight "$SAFE_LEARNED" \
      --candidate_dta_safe_mix_gate_limit 1.0 \
      --candidate_dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
      "${ROUTER_EVAL_FLAG[@]}" \
      --depth_shuffle_offset 137 \
      --split_json "$SPLIT_JSON" \
      --split_name "$EVAL_SPLIT" \
      --eval_root_split train \
      --output_dir "$COMPARE_DIR" \
      --tag "$EVAL_RUN" \
      --max_images "$MAX_IMAGES" \
      2>&1 | tee "$EVID/dta_v3_3_${EVAL_RUN}_eval.log"
    eval_rc=${PIPESTATUS[0]}
    set -e
    echo "dta_v3_3_routerfusion_eval_done rc=$eval_rc run_id=$EVAL_RUN $(date --iso-8601=seconds)" | tee -a "$STATUS"
    if [[ "$eval_rc" -ne 0 ]]; then exit "$eval_rc"; fi
    if [[ "$first" -eq 0 ]]; then printf ',\n' >> "$MANIFEST"; fi
    first=0
    label="$EVAL_MODE"
    if [[ "$EVAL_MODE" == "invert" ]]; then label="true"; fi
    printf '  {"label":"%s","train_depth":"invert","eval_depth":"%s","airlight_mode":"%s","compare_dir":"%s"}' "$label" "$EVAL_MODE" "$A_MODE" "$COMPARE_DIR" >> "$MANIFEST"
  done
  printf '\n]}\n' >> "$MANIFEST"
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/aggregate_haze4k_dta_v3_controls.py \
    --manifest "$MANIFEST" \
    --output_matrix_json "$EVID/train_eval_depth_matrix_${RUN_ID}_${A_MODE}.json" \
    --output_matrix_csv "$EVID/train_eval_depth_matrix_${RUN_ID}_${A_MODE}.csv" \
    --output_attribution_csv "$EVID/r0_vs_rdepth_attribution_${RUN_ID}_${A_MODE}.csv" \
    --baseline_label zero \
    --true_label true \
    2>&1 | tee "$EVID/dta_v3_3_${RUN_ID}_${A_MODE}_aggregate.log"
  agg_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_3_routerfusion_aggregate_done rc=$agg_rc run_id=$RUN_ID airlight=$A_MODE $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$agg_rc" -ne 0 ]]; then exit "$agg_rc"; fi
done

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/summarize_haze4k_dta_v33_routerfusion_scouts.py \
  --evidence_dir "$EVID" \
  --output_json "$EVID/dta_v3_3_routerfusion_triage_summary.json" \
  --output_csv "$EVID/dta_v3_3_routerfusion_triage_summary.csv" \
  --variant_csv "$EVID/dta_v3_3_routerfusion_variant_summary.csv" \
  2>&1 | tee "$EVID/dta_v3_3_${RUN_ID}_summary.log"
summary_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_3_routerfusion_summary_done rc=$summary_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$summary_rc" -ne 0 ]]; then exit "$summary_rc"; fi

if [[ "$RUN_DIAG" == "1" ]]; then
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_v32_action_diagnostics.py \
    --run_id "$RUN_ID" \
    --candidate_name "$VARIANT" \
    --data_dir "$DATA" \
    --a0_checkpoint "$A0" \
    --candidate_checkpoint "$CANDIDATE" \
    --depth_cache_dir "$DEPTH" \
    --split_json "$SPLIT_JSON" \
    --split_name "$EVAL_SPLIT" \
    --eval_root_split train \
    --depth_split train \
    --output_dir "$EVID" \
    --max_images "$MAX_IMAGES" \
    --depth_modes invert,zero,shuffle,normal \
    --airlight_modes fallback,gt \
    --dta_prior_channels 32 --dta_gate_bias -5.0 --dta_gate_limit 0.18 --dta_gamma_limit 0.28 --dta_beta_limit 0.14 \
    --dta_confidence_floor 0.30 --dta_r0_residual_scale 0.0 --dta_depth_residual_scale 0.08 \
    --dta_depth_mask_easy_budget 0.04 --dta_depth_mask_dense_budget 0.14 --dta_depth_mask_bias -4.0 --dta_phys_t_min 0.10 \
    --dta_phase depth --dta_ablation full --dta_safe_mix_enabled --dta_safe_mix_delta_clip "$SAFE_CLIP" \
    --dta_safe_mix_phys_weight "$SAFE_PHYS" --dta_safe_mix_learned_weight "$SAFE_LEARNED" \
    --dta_safe_mix_gate_limit 1.0 --dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
    "${ROUTER_FLAG[@]}" \
    2>&1 | tee "$EVID/dta_v3_3_${RUN_ID}_router_diagnostics.log"
  diag_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_3_routerfusion_diag_done rc=$diag_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$diag_rc" -ne 0 ]]; then exit "$diag_rc"; fi
fi

CONTACT_DIR=$EVID/tail_regression_contact_sheet/${RUN_ID}_fallback
mkdir -p "$CONTACT_DIR"
true_csv=$(ls "$EVID"/dta_v3_3_${RUN_ID}_fallback_invert_compare/scout_eval_per_image_*.csv | head -n 1)
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_contact_sheet.py \
  --data_dir "$DATA" --depth_cache_dir "$DEPTH" --depth_split train --root_split train \
  --split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT" \
  --per_image_csv "$true_csv" \
  --a0_checkpoint "$A0" --candidate_checkpoint "$CANDIDATE" --candidate_arch dta_v3 \
  --output_dir "$CONTACT_DIR" --tag "$RUN_ID" --count 12 \
  --dta_variant v3 --dta_depth_mode invert --dta_airlight_mode fallback --dta_phase depth --dta_ablation full \
  --dta_prior_channels 32 --dta_gate_bias -5.0 --dta_gate_limit 0.18 --dta_gamma_limit 0.28 --dta_beta_limit 0.14 \
  --dta_confidence_floor 0.30 --dta_r0_residual_scale 0.0 --dta_depth_residual_scale 0.08 \
  --dta_depth_mask_easy_budget 0.04 --dta_depth_mask_dense_budget 0.14 --dta_depth_mask_bias -4.0 \
  --dta_safe_mix_enabled --dta_safe_mix_delta_clip "$SAFE_CLIP" --dta_safe_mix_phys_weight "$SAFE_PHYS" \
  --dta_safe_mix_learned_weight "$SAFE_LEARNED" --dta_safe_mix_gate_limit 1.0 --dta_safe_mix_gate_bias "$SAFE_GATE_BIAS" \
  "${ROUTER_FLAG[@]}" \
  2>&1 | tee "$EVID/dta_v3_3_${RUN_ID}_contact_sheet.log"
contact_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_3_routerfusion_contact_sheet_done rc=$contact_rc run_id=$RUN_ID output=$CONTACT_DIR $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$contact_rc" -ne 0 ]]; then exit "$contact_rc"; fi

echo "DTA_V3_3_ROUTERFUSION_SCOUT_OK run_id=$RUN_ID checkpoint=$CANDIDATE" | tee -a "$STATUS"
