#!/usr/bin/env bash
set -euo pipefail
ABLATION=${1:-film_only_no_output_refine}
STAGE=${2:-scout5}
SEED=${3:-3407}
FOLD=${4:-0}
DEPTH_MODE=${5:-invert}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune}
ITS=$WORK/Dehazing/ITS
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
TRAIN_SPLIT=${TRAIN_SPLIT:-fold${FOLD}_train}
EVAL_SPLIT=${EVAL_SPLIT:-fold${FOLD}_val}
ROUTE_INIT=${ROUTE_INIT:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
case "$STAGE" in
  scout5) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1; MAX_IMAGES=${MAX_IMAGES:-128} ;;
  oof20) NUM_EPOCH=20; STOP_EPOCH=20; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=5; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac
case "$ABLATION" in
  r0_only) PHASE=r0; TRAIN_SCOPE=dta_r0_only; INIT_CKPT=$A0; INIT_ARGS=(--init_model_partial --partial_new_prefixes DTA.); DEPTH_MODE=zero ;;
  film_only_no_output_refine) PHASE=depth; TRAIN_SCOPE=dta_depth_only; INIT_CKPT=$A0; INIT_ARGS=(--init_model_partial --partial_new_prefixes DTA.) ;;
  trans_head_only_no_rgb_residual) PHASE=depth; TRAIN_SCOPE=dta_depth_only; INIT_CKPT=$A0; INIT_ARGS=(--init_model_partial --partial_new_prefixes DTA.) ;;
  phys_blend_only) PHASE=depth; TRAIN_SCOPE=dta_depth_only; INIT_CKPT=${ROUTE_INIT:-$A0}; if [[ -n "$ROUTE_INIT" ]]; then INIT_ARGS=(--init_model_allow_full_route); else INIT_ARGS=(--init_model_partial --partial_new_prefixes DTA.); fi ;;
  *) echo "Unsupported ABLATION=$ABLATION" >&2; exit 64 ;;
esac
RUN_ID=${STAGE}_ablation_${ABLATION}_${DEPTH_MODE}_seed${SEED}_f${FOLD}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-DAPC-Ablation-${ABLATION}-${DEPTH_MODE}-seed${SEED}-f${FOLD}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_${RUN_ID}_train.log
EVAL_LOG=$EVID/dta_v3_${RUN_ID}_eval.log
COMPARE_DIR=$EVID/dta_v3_${RUN_ID}_compare
TPRED_DIR=$EVID/dta_v3_${RUN_ID}_tpred
mkdir -p "$COMPARE_DIR" "$TPRED_DIR"
{
  echo "ablation_start run_id=$RUN_ID $(date --iso-8601=seconds)"
  echo "ablation=$ABLATION phase=$PHASE train_scope=$TRAIN_SCOPE init=$INIT_CKPT depth_mode=$DEPTH_MODE"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"
cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" --data Haze4K --version base --fam_mode original \
  --arch dta_v3 --dta_variant v3 --seed "$SEED" --mode train --data_dir "$DATA" \
  --batch_size 4 --learning_rate 0.0001 --weight_decay 0.0001 \
  --num_epoch "$NUM_EPOCH" --stop_epoch "$STOP_EPOCH" --print_freq 50 --num_worker 4 \
  --save_freq "$SAVE_FREQ" --valid_freq "$VALID_FREQ" --valid_root_split train \
  --mod_stats_freq "$MOD_STATS_FREQ" --mod_stats_batches 16 \
  --grad_clip_norm 0.001 --dta_grad_clip_norm 0.05 \
  --init_model "$INIT_CKPT" "${INIT_ARGS[@]}" \
  --train_scope "$TRAIN_SCOPE" --dta_depth_cache_dir "$DEPTH" \
  --dta_train_depth_split train --dta_eval_depth_split train --dta_require_depth \
  --dta_depth_mode "$DEPTH_MODE" --dta_phase "$PHASE" --dta_ablation "$ABLATION" \
  --dta_prior_channels 32 --dta_gate_bias -5.0 --dta_gate_limit 0.10 --dta_gamma_limit 0.16 --dta_beta_limit 0.08 \
  --dta_confidence_floor 0.30 --dta_r0_residual_scale 0.04 --dta_depth_residual_scale 0.08 \
  --dta_depth_mask_easy_budget 0.04 --dta_depth_mask_dense_budget 0.12 --dta_depth_mask_bias -4.0 --dta_phys_t_min 0.10 \
  --dta_use_trans_gt --dta_rank_weight 0.001 --dta_tv_weight 0.0001 \
  --dta_trans_weight 0.02 --dta_phys_weight 0.005 --dta_preserve_weight 0.02 \
  --dta_reference_checkpoint "$A0" --dta_ref_preserve_weight 0.02 --dta_tail_guard_weight 0.02 \
  --split_json "$SPLIT_JSON" --split_name "$TRAIN_SPLIT" \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "ablation_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
CANDIDATE=$ITS/results/$MODEL_NAME/Training-Results/Final.pkl
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
  --data_dir "$DATA" --original_checkpoint "$A0" --original_arch official_convir --original_name A0 \
  --candidate_checkpoint "$CANDIDATE" --candidate_arch dta_v3 --candidate_name "DTA_v3_${RUN_ID}" \
  --dta_depth_cache_dir "$DEPTH" --dta_eval_depth_split train \
  --candidate_dta_variant v3 --candidate_dta_depth_mode "$DEPTH_MODE" --candidate_dta_phase "$PHASE" --candidate_dta_ablation "$ABLATION" \
  --candidate_dta_prior_channels 32 --candidate_dta_gate_bias -5.0 --candidate_dta_gate_limit 0.10 --candidate_dta_gamma_limit 0.16 --candidate_dta_beta_limit 0.08 \
  --candidate_dta_confidence_floor 0.30 --candidate_dta_r0_residual_scale 0.04 --candidate_dta_depth_residual_scale 0.08 \
  --candidate_dta_depth_mask_easy_budget 0.04 --candidate_dta_depth_mask_dense_budget 0.12 --candidate_dta_depth_mask_bias -4.0 \
  --split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT" --eval_root_split train \
  --output_dir "$COMPARE_DIR" --tag "$RUN_ID" --max_images "$MAX_IMAGES" \
  2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
set -e
echo "ablation_eval_done rc=$eval_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$eval_rc" -ne 0 ]]; then exit "$eval_rc"; fi
cp "$COMPARE_DIR"/scout_eval_compare_*.json "$EVID/${ABLATION}.json" 2>/dev/null || true
echo "DTA_V3_ABLATION_OK run_id=$RUN_ID" | tee -a "$STATUS"
