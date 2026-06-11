#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-scout5full}
SEED=${2:-3407}
FOLD=${3:-0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v31}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
TRAIN_SPLIT=${TRAIN_SPLIT:-fold${FOLD}_train}
EVAL_SPLIT=${EVAL_SPLIT:-fold${FOLD}_val}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "$STAGE" in
  smoke) NUM_EPOCH=1; STOP_EPOCH=1; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  scout5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1 ;;
  scout10full) NUM_EPOCH=10; STOP_EPOCH=10; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=2 ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac

RUN_ID=v31_wg18_light_hinge_seed${SEED}_f${FOLD}_${STAGE}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-1-WG18LightHinge-seed${SEED}-f${FOLD}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_1_${RUN_ID}_train.log
mkdir -p "$EVID"
{
  echo "dta_v3_1_light_hinge_train_start run_id=$RUN_ID stage=$STAGE $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "train_split=$TRAIN_SPLIT"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "params wg18 gate=0.18 gamma=0.28 beta=0.14 depth_scale=0.08 dense_budget=0.14 preserve=0.03 ref=0.02 light_tail=0.03 light_ssim=0.01"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
cd "$WORK/Dehazing/ITS"
if [[ "${FORCE:-0}" != "1" && -f "results/$MODEL_NAME/Training-Results/Final.pkl" ]]; then
  echo "dta_v3_1_light_hinge_train_skip_existing model=$MODEL_NAME $(date --iso-8601=seconds)" | tee -a "$STATUS"
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
    --learning_rate 0.00005 \
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
    --train_scope dta_depth_only \
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
    --dta_gate_ramp_start 0.12 \
    --dta_gate_ramp_mid 0.18 \
    --dta_gate_ramp_end 0.18 \
    --dta_gate_ramp_warmup_epochs 1 \
    --dta_gate_ramp_mid_epochs 3 \
    --dta_confidence_floor 0.30 \
    --dta_r0_residual_scale 0.0 \
    --dta_depth_residual_scale 0.08 \
    --dta_depth_mask_easy_budget 0.04 \
    --dta_depth_mask_dense_budget 0.14 \
    --dta_depth_mask_density_thresh 0.35 \
    --dta_depth_mask_bias -4.0 \
    --dta_phys_t_min 0.10 \
    --dta_use_trans_gt \
    --dta_rank_weight 0.001 \
    --dta_tv_weight 0.0001 \
    --dta_proxy_weight 0.0 \
    --dta_trans_weight 0.02 \
    --dta_phys_weight 0.005 \
    --dta_preserve_weight 0.03 \
    --dta_preserve_trans_thresh 0.80 \
    --dta_reference_checkpoint "$A0" \
    --dta_ref_preserve_weight 0.02 \
    --dta_tail_guard_weight 0.0 \
    --dta_tail_guard_margin 0.0 \
    --dta_mask_budget_weight 0.001 \
    --dta_light_tail_hinge_weight 0.03 \
    --dta_light_tail_hinge_margin 0.0 \
    --dta_light_tail_hinge_topk 0.10 \
    --dta_light_hinge_bright_thresh 0.82 \
    --dta_light_hinge_texture_thresh 0.004 \
    --dta_light_ssim_hinge_weight 0.01 \
    --dta_light_ssim_hinge_margin 0.0 \
    --split_json "$SPLIT_JSON" \
    --split_name "$TRAIN_SPLIT" \
    2>&1 | tee "$TRAIN_LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_1_light_hinge_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
fi

CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_1_LIGHT_HINGE_MISSING_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

cd "$WORK"
RUN_ID=${RUN_ID}_post CANDIDATE="$CANDIDATE" EVAL_SPLIT="$EVAL_SPLIT" MAX_IMAGES=${MAX_IMAGES:-0} \
  bash "$EVID/run_dta_v3_1_wg18_riskselect_audit_convir4090.sh"
echo "DTA_V3_1_WG18_LIGHT_HINGE_SCOUT_OK run_id=$RUN_ID checkpoint=$CANDIDATE" | tee -a "$STATUS"
