#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-scout5}
SCOPE=${2:-adapter_only}
DEPTH_MODE=${3:-normal}
SEED=${4:-3407}
FOLD_TAG=${FOLD_TAG:-nosplit}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-v2-calibrated}
ITS=$REMOTE_ROOT/Dehazing/ITS
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_v2_calibrated_20260611
STATUS=$EVID/status.txt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

SPLIT_JSON=${SPLIT_JSON:-}
TRAIN_SPLIT=${TRAIN_SPLIT:-}
EVAL_SPLIT=${EVAL_SPLIT:-}
USER_VALID_FREQ=${VALID_FREQ:-}
USER_EVAL_ROOT_SPLIT=${EVAL_ROOT_SPLIT:-}
USER_EVAL_DEPTH_SPLIT=${EVAL_DEPTH_SPLIT:-}
USER_VALID_ROOT_SPLIT=${VALID_ROOT_SPLIT:-}
EVAL_ROOT_SPLIT=${EVAL_ROOT_SPLIT:-test}
TRAIN_DEPTH_SPLIT=${TRAIN_DEPTH_SPLIT:-train}
EVAL_DEPTH_SPLIT=${EVAL_DEPTH_SPLIT:-test}
VALID_ROOT_SPLIT=${VALID_ROOT_SPLIT:-test}

case "$STAGE" in
  smoke)
    NUM_EPOCH=1
    STOP_EPOCH=1
    SAVE_FREQ=1
    VALID_FREQ=${VALID_FREQ:-1}
    MOD_STATS_FREQ=1
    MAX_IMAGES=${MAX_IMAGES:-32}
    ;;
  scout5)
    NUM_EPOCH=5
    STOP_EPOCH=5
    SAVE_FREQ=1
    VALID_FREQ=${VALID_FREQ:-1}
    MOD_STATS_FREQ=1
    MAX_IMAGES=${MAX_IMAGES:-128}
    ;;
  gate20|oof20)
    NUM_EPOCH=20
    STOP_EPOCH=20
    SAVE_FREQ=5
    VALID_FREQ=${VALID_FREQ:-5}
    MOD_STATS_FREQ=5
    MAX_IMAGES=${MAX_IMAGES:-0}
    ;;
  *)
    echo "Unsupported STAGE=$STAGE; expected smoke, scout5, gate20, or oof20" >&2
    exit 64
    ;;
esac

if [[ -n "$SPLIT_JSON" ]]; then
  if [[ -z "$TRAIN_SPLIT" || -z "$EVAL_SPLIT" ]]; then
    echo "SPLIT_JSON requires TRAIN_SPLIT and EVAL_SPLIT" >&2
    exit 64
  fi
  if [[ -z "$USER_VALID_FREQ" ]]; then
    VALID_FREQ=9999
  fi
  if [[ -z "$USER_EVAL_ROOT_SPLIT" ]]; then
    EVAL_ROOT_SPLIT=train
  fi
  if [[ -z "$USER_VALID_ROOT_SPLIT" ]]; then
    VALID_ROOT_SPLIT=train
  fi
  if [[ -z "$USER_EVAL_DEPTH_SPLIT" ]]; then
    EVAL_DEPTH_SPLIT=train
  fi
fi

MODEL_NAME=ConvIR-Haze4K-DTA-v2-${SCOPE}-${DEPTH_MODE}-seed${SEED}-${FOLD_TAG}-${STAGE}
RUN_ID=${STAGE}_${SCOPE}_${DEPTH_MODE}_seed${SEED}_${FOLD_TAG}
TRAIN_LOG=$EVID/dta_v2_${RUN_ID}_train.log
EVAL_LOG=$EVID/dta_v2_${RUN_ID}_eval.log
COMPARE_DIR=$EVID/dta_v2_${RUN_ID}_compare
TPRED_LOG=$EVID/dta_v2_${RUN_ID}_tpred.log
TPRED_DIR=$EVID/dta_v2_${RUN_ID}_tpred
mkdir -p "$EVID" "$COMPARE_DIR" "$TPRED_DIR"

{
  echo "train_eval_start dta_v2 run_id=$RUN_ID $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "depth=$DEPTH"
  echo "model_name=$MODEL_NAME"
  echo "stage=$STAGE"
  echo "scope=$SCOPE"
  echo "depth_mode=$DEPTH_MODE"
  echo "seed=$SEED"
  echo "split_json=$SPLIT_JSON"
  echo "train_split=$TRAIN_SPLIT"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"

cd "$REMOTE_ROOT"
{
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
} | tee -a "$STATUS"

TRAIN_SPLIT_ARGS=()
if [[ -n "$SPLIT_JSON" ]]; then
  TRAIN_SPLIT_ARGS=(--split_json "$SPLIT_JSON" --split_name "$TRAIN_SPLIT")
fi

cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch dta_v2 \
  --dta_variant v2 \
  --seed "$SEED" \
  --mode train \
  --data_dir "$DATA" \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --weight_decay 0 \
  --num_epoch "$NUM_EPOCH" \
  --stop_epoch "$STOP_EPOCH" \
  --print_freq 50 \
  --num_worker 4 \
  --save_freq "$SAVE_FREQ" \
  --valid_freq "$VALID_FREQ" \
  --valid_root_split "$VALID_ROOT_SPLIT" \
  --mod_stats_freq "$MOD_STATS_FREQ" \
  --mod_stats_batches 16 \
  --grad_clip_norm 0.001 \
  --dta_grad_clip_norm 0.05 \
  --dta_neighbor_grad_clip_norm 0.005 \
  --init_model "$A0" \
  --init_model_partial \
  --partial_new_prefixes DTA. \
  --train_scope "$SCOPE" \
  --dta_depth_cache_dir "$DEPTH" \
  --dta_train_depth_split "$TRAIN_DEPTH_SPLIT" \
  --dta_eval_depth_split "$EVAL_DEPTH_SPLIT" \
  --dta_require_depth \
  --dta_depth_mode "$DEPTH_MODE" \
  --dta_prior_channels 32 \
  --dta_gate_bias -6.0 \
  --dta_gate_limit 0.06 \
  --dta_gamma_limit 0.12 \
  --dta_beta_limit 0.06 \
  --dta_alpha_init 1.0 \
  --dta_confidence_floor 0.25 \
  --dta_confidence_local_scale 6.0 \
  --dta_output_residual_scale 0.03 \
  --dta_use_trans_gt \
  --dta_rank_weight 0.001 \
  --dta_tv_weight 0.0001 \
  --dta_proxy_weight 0.0 \
  --dta_trans_weight 0.02 \
  --dta_phys_weight 0.005 \
  --dta_preserve_weight 0.02 \
  --dta_preserve_trans_thresh 0.80 \
  --dta_gate_ramp_start 0.01 \
  --dta_gate_ramp_mid 0.03 \
  --dta_gate_ramp_end 0.06 \
  --dta_gate_ramp_warmup_epochs 2 \
  --dta_gate_ramp_mid_epochs 8 \
  "${TRAIN_SPLIT_ARGS[@]}" \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "train_done rc=$train_rc dta_v2 run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then
  echo "DTA_V2_TRAIN_FAILED run_id=$RUN_ID" | tee -a "$STATUS"
  exit "$train_rc"
fi

CANDIDATE=$ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then
  echo "FAILED_INFRA_MISSING_CHECKPOINT run_id=$RUN_ID path=$CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

COMPARE_SPLIT_ARGS=()
if [[ -n "$SPLIT_JSON" ]]; then
  COMPARE_SPLIT_ARGS=(--split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT" --eval_root_split "$EVAL_ROOT_SPLIT")
else
  COMPARE_SPLIT_ARGS=(--eval_root_split "$EVAL_ROOT_SPLIT")
fi

cd "$REMOTE_ROOT"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
  --data_dir "$DATA" \
  --original_checkpoint "$A0" \
  --original_arch official_convir \
  --original_mode original \
  --original_name A0 \
  --candidate_checkpoint "$CANDIDATE" \
  --candidate_arch dta_v2 \
  --candidate_mode original \
  --candidate_name "DTA_v2_${RUN_ID}_final" \
  --dta_depth_cache_dir "$DEPTH" \
  --dta_eval_depth_split "$EVAL_DEPTH_SPLIT" \
  --candidate_dta_variant v2 \
  --candidate_dta_depth_mode "$DEPTH_MODE" \
  --candidate_dta_prior_channels 32 \
  --candidate_dta_gate_bias -6.0 \
  --candidate_dta_gate_limit 0.06 \
  --candidate_dta_gamma_limit 0.12 \
  --candidate_dta_beta_limit 0.06 \
  --candidate_dta_alpha_init 1.0 \
  --candidate_dta_confidence_floor 0.25 \
  --candidate_dta_confidence_local_scale 6.0 \
  --candidate_dta_output_residual_scale 0.03 \
  --output_dir "$COMPARE_DIR" \
  --tag "$RUN_ID" \
  --max_images "$MAX_IMAGES" \
  "${COMPARE_SPLIT_ARGS[@]}" \
  2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
set -e
echo "eval_done rc=$eval_rc dta_v2 run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$eval_rc" -eq 0 ]]; then
  echo "DTA_V2_TRAIN_EVAL_OK run_id=$RUN_ID" | tee -a "$STATUS"
else
  echo "DTA_V2_EVAL_FAILED run_id=$RUN_ID" | tee -a "$STATUS"
fi
if [[ "$eval_rc" -ne 0 ]]; then
  exit "$eval_rc"
fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_v2_checkpoint.py \
  --checkpoint "$CANDIDATE" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split "$EVAL_DEPTH_SPLIT" \
  --dta_depth_mode "$DEPTH_MODE" \
  --dta_prior_channels 32 \
  --dta_gate_bias -6.0 \
  --dta_gate_limit 0.06 \
  --dta_gamma_limit 0.12 \
  --dta_beta_limit 0.06 \
  --dta_alpha_init 1.0 \
  --dta_confidence_floor 0.25 \
  --dta_confidence_local_scale 6.0 \
  --dta_output_residual_scale 0.03 \
  --output_dir "$TPRED_DIR" \
  --tag "$RUN_ID" \
  --max_images "$MAX_IMAGES" \
  "${COMPARE_SPLIT_ARGS[@]}" \
  2>&1 | tee "$TPRED_LOG"
tpred_rc=${PIPESTATUS[0]}
set -e
echo "tpred_done rc=$tpred_rc dta_v2 run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$tpred_rc" -eq 0 ]]; then
  echo "DTA_V2_TPRED_AUDIT_OK run_id=$RUN_ID" | tee -a "$STATUS"
else
  echo "DTA_V2_TPRED_AUDIT_FAILED run_id=$RUN_ID" | tee -a "$STATUS"
fi
exit "$tpred_rc"
