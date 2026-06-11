#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:-scout5full}
VARIANT=${2:-r0s010_lr3e5_ref005}
SEED=${3:-3407}
FOLD=${4:-0}
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "$STAGE" in
  smoke) NUM_EPOCH=1; STOP_EPOCH=1; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1; MAX_IMAGES=${MAX_IMAGES:-64} ;;
  scout5) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1; MAX_IMAGES=${MAX_IMAGES:-128} ;;
  scout5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  scout10full) NUM_EPOCH=10; STOP_EPOCH=10; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=2; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  oof20) NUM_EPOCH=20; STOP_EPOCH=20; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=5; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac

case "$VARIANT" in
  r0s005_lr3e5_ref005)
    LR=0.00003; R0_SCALE=0.005; PRESERVE_W=0.05; REF_W=0.05; TAIL_W=0.05; GRAD_CLIP=0.0005; DTA_GRAD_CLIP=0.02 ;;
  r0s010_lr3e5_ref005)
    LR=0.00003; R0_SCALE=0.010; PRESERVE_W=0.05; REF_W=0.05; TAIL_W=0.05; GRAD_CLIP=0.0005; DTA_GRAD_CLIP=0.02 ;;
  r0s020_lr3e5_ref005)
    LR=0.00003; R0_SCALE=0.020; PRESERVE_W=0.05; REF_W=0.05; TAIL_W=0.05; GRAD_CLIP=0.0005; DTA_GRAD_CLIP=0.02 ;;
  r0s010_lr1e5_ref010)
    LR=0.00001; R0_SCALE=0.010; PRESERVE_W=0.10; REF_W=0.10; TAIL_W=0.10; GRAD_CLIP=0.00025; DTA_GRAD_CLIP=0.01 ;;
  *) echo "Unsupported VARIANT=$VARIANT" >&2; exit 65 ;;
esac

RUN_ID=${STAGE}_phaseA_${VARIANT}_seed${SEED}_f${FOLD}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-DAPC-PhaseA-${VARIANT}-seed${SEED}-f${FOLD}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_${RUN_ID}_train.log
EVAL_LOG=$EVID/dta_v3_${RUN_ID}_eval.log
COMPARE_DIR=$EVID/dta_v3_${RUN_ID}_compare
CONTACT_DIR=$EVID/tail_regression_contact_sheet/$RUN_ID
CONTACT_LOG=$EVID/dta_v3_${RUN_ID}_contact_sheet.log
mkdir -p "$EVID" "$COMPARE_DIR" "$CONTACT_DIR"
{
  echo "phase_a_variant_start run_id=$RUN_ID variant=$VARIANT stage=$STAGE $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "train_split=$TRAIN_SPLIT"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "lr=$LR r0_scale=$R0_SCALE preserve=$PRESERVE_W ref=$REF_W tail=$TAIL_W grad_clip=$GRAD_CLIP dta_grad_clip=$DTA_GRAD_CLIP"
} | tee -a "$STATUS"
cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
cd "$ITS"
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
  --grad_clip_norm "$GRAD_CLIP" \
  --dta_grad_clip_norm "$DTA_GRAD_CLIP" \
  --init_model "$A0" \
  --init_model_partial \
  --partial_new_prefixes DTA. \
  --train_scope dta_r0_only \
  --dta_depth_cache_dir "$DEPTH" \
  --dta_train_depth_split train \
  --dta_eval_depth_split train \
  --dta_require_depth \
  --dta_depth_mode zero \
  --dta_phase r0 \
  --dta_ablation r0_only \
  --dta_prior_channels 32 \
  --dta_gate_bias -5.0 \
  --dta_gate_limit 0.10 \
  --dta_gamma_limit 0.16 \
  --dta_beta_limit 0.08 \
  --dta_confidence_floor 0.30 \
  --dta_r0_residual_scale "$R0_SCALE" \
  --dta_use_trans_gt \
  --dta_rank_weight 0.0 \
  --dta_tv_weight 0.0 \
  --dta_trans_weight 0.0 \
  --dta_phys_weight 0.0 \
  --dta_preserve_weight "$PRESERVE_W" \
  --dta_preserve_trans_thresh 0.80 \
  --dta_reference_checkpoint "$A0" \
  --dta_ref_preserve_weight "$REF_W" \
  --dta_tail_guard_weight "$TAIL_W" \
  --dta_tail_guard_margin 0.0 \
  --split_json "$SPLIT_JSON" \
  --split_name "$TRAIN_SPLIT" \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "phase_a_variant_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
CANDIDATE=$ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then echo "MISSING_PHASE_A_VARIANT_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"; exit 3; fi
cd "$WORK"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
  --data_dir "$DATA" \
  --original_checkpoint "$A0" \
  --original_arch official_convir \
  --original_name A0 \
  --candidate_checkpoint "$CANDIDATE" \
  --candidate_arch dta_v3 \
  --candidate_name "DTA_v3_PhaseA_${VARIANT}" \
  --dta_depth_cache_dir "$DEPTH" \
  --dta_eval_depth_split train \
  --candidate_dta_variant v3 \
  --candidate_dta_depth_mode zero \
  --candidate_dta_phase r0 \
  --candidate_dta_ablation r0_only \
  --candidate_dta_prior_channels 32 \
  --candidate_dta_gate_bias -5.0 \
  --candidate_dta_gate_limit 0.10 \
  --candidate_dta_gamma_limit 0.16 \
  --candidate_dta_beta_limit 0.08 \
  --candidate_dta_confidence_floor 0.30 \
  --candidate_dta_r0_residual_scale "$R0_SCALE" \
  --split_json "$SPLIT_JSON" \
  --split_name "$EVAL_SPLIT" \
  --eval_root_split train \
  --output_dir "$COMPARE_DIR" \
  --tag "$RUN_ID" \
  --max_images "$MAX_IMAGES" \
  2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
set -e
echo "phase_a_variant_eval_done rc=$eval_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$eval_rc" -ne 0 ]]; then exit "$eval_rc"; fi
PER_IMAGE=$COMPARE_DIR/scout_eval_per_image_${RUN_ID}.csv
if [[ "${MAKE_CONTACTSHEETS:-1}" == "1" && -f "$PER_IMAGE" ]]; then
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_contact_sheet.py \
    --data_dir "$DATA" \
    --depth_cache_dir "$DEPTH" \
    --depth_split train \
    --root_split train \
    --split_json "$SPLIT_JSON" \
    --split_name "$EVAL_SPLIT" \
    --per_image_csv "$PER_IMAGE" \
    --a0_checkpoint "$A0" \
    --candidate_checkpoint "$CANDIDATE" \
    --candidate_arch dta_v3 \
    --output_dir "$CONTACT_DIR" \
    --tag "$RUN_ID" \
    --count 12 \
    --dta_variant v3 \
    --dta_depth_mode zero \
    --dta_phase r0 \
    --dta_ablation r0_only \
    --dta_prior_channels 32 \
    --dta_gate_bias -5.0 \
    --dta_gate_limit 0.10 \
    --dta_gamma_limit 0.16 \
    --dta_beta_limit 0.08 \
    --dta_confidence_floor 0.30 \
    --dta_r0_residual_scale "$R0_SCALE" \
    2>&1 | tee "$CONTACT_LOG"
  contact_rc=${PIPESTATUS[0]}
  set -e
  echo "phase_a_variant_contact_sheet_done rc=$contact_rc run_id=$RUN_ID output=$CONTACT_DIR $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$contact_rc" -ne 0 ]]; then exit "$contact_rc"; fi
else
  echo "phase_a_variant_contact_sheet_skipped run_id=$RUN_ID make=${MAKE_CONTACTSHEETS:-1} per_image=$PER_IMAGE" | tee -a "$STATUS"
fi
echo "DTA_V3_PHASE_A_R0_VARIANT_EVAL_COMPLETE run_id=$RUN_ID checkpoint=$CANDIDATE compare=$COMPARE_DIR contact_sheet=$CONTACT_DIR" | tee -a "$STATUS"
