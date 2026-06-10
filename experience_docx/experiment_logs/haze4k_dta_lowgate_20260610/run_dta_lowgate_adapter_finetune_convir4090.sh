#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-scout5}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
REMOTE_ROOT=${REMOTE_ROOT:-$BASE/repos/ConvIR-B-dta-lowgate}
ITS=$REMOTE_ROOT/Dehazing/ITS
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/haze4k_dta_lowgate_20260610
PY=$BASE/envs/convir-cu121/bin/python
DATA=$BASE/datasets/Haze4K/Haze4K
A0=$BASE/checkpoints/official/Haze4K/haze4k-base.pkl
STATUS=$EVID/status.txt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

case "$STAGE" in
  scout5)
    MODEL_NAME=ConvIR-Haze4K-DTA-lowgate-adapteronly-seed3407-20260611-scout5
    NUM_EPOCH=5
    STOP_EPOCH=5
    SAVE_FREQ=1
    VALID_FREQ=1
    MOD_STATS_FREQ=1
    MAX_IMAGES=128
    TAG=scout5_seed3407_max128
    ;;
  gate20)
    MODEL_NAME=ConvIR-Haze4K-DTA-lowgate-adapteronly-seed3407-20260611-gate20
    NUM_EPOCH=20
    STOP_EPOCH=20
    SAVE_FREQ=5
    VALID_FREQ=5
    MOD_STATS_FREQ=5
    MAX_IMAGES=0
    TAG=gate20_seed3407_full
    ;;
  *)
    echo "Unsupported STAGE=$STAGE; expected scout5 or gate20" >&2
    exit 64
    ;;
esac

TRAIN_LOG=$EVID/dta_lowgate_${STAGE}_train.log
EVAL_LOG=$EVID/dta_lowgate_${STAGE}_eval.log
COMPARE_DIR=$EVID/dta_lowgate_${STAGE}_compare

find_depth_cache() {
  if [[ -n "${DEPTH:-}" && -d "${DEPTH:-}" ]]; then
    printf '%s\n' "$DEPTH"
    return 0
  fi
  local candidates=(
    "$BASE/depth_cache/depth_anything_v2_small_hf"
    "$BASE/caches/Haze4K/depth_anything_v2_small_hf"
    "$BASE/datasets/Haze4K/depth_anything_v2_small_hf"
    "$BASE/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf"
    "/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf"
  )
  local item
  for item in "${candidates[@]}"; do
    if [[ -d "$item/train" && -d "$item/test" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  done
  return 1
}

mkdir -p "$EVID" "$COMPARE_DIR"
{
  echo "${STAGE}_start haze4k_dta_lowgate $(date --iso-8601=seconds)"
  echo "remote_root=$REMOTE_ROOT"
  echo "python=$PY"
  echo "data=$DATA"
  echo "checkpoint=$A0"
  echo "model_name=$MODEL_NAME"
  echo "num_epoch=$NUM_EPOCH"
  echo "stop_epoch=$STOP_EPOCH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"

if ! DEPTH_CACHE=$(find_depth_cache); then
  echo "FAILED_INFRA_MISSING_DEPTH_CACHE stage=$STAGE" | tee -a "$STATUS"
  exit 2
fi
echo "depth_cache=$DEPTH_CACHE" | tee -a "$STATUS"

cd "$REMOTE_ROOT"
{
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
} | tee -a "$STATUS"

cd "$ITS"
set +e
PYTHONUNBUFFERED=1 "$PY" main.py \
  --model_name "$MODEL_NAME" \
  --data Haze4K \
  --version base \
  --fam_mode original \
  --arch dta \
  --seed 3407 \
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
  --mod_stats_freq "$MOD_STATS_FREQ" \
  --mod_stats_batches 16 \
  --grad_clip_norm 0.001 \
  --init_model "$A0" \
  --init_model_partial \
  --partial_new_prefixes DTA. \
  --train_scope adapter_only \
  --dta_depth_cache_dir "$DEPTH_CACHE" \
  --dta_train_depth_split train \
  --dta_eval_depth_split test \
  --dta_require_depth \
  --dta_gate_bias -7.0 \
  --dta_gate_limit 0.03 \
  --dta_gamma_limit 0.10 \
  --dta_beta_limit 0.05 \
  --dta_rank_weight 0.003 \
  --dta_tv_weight 0.0003 \
  --dta_proxy_weight 0.0 \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "${STAGE}_train_done rc=$train_rc haze4k_dta_lowgate $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then
  echo "DTA_LOWGATE_${STAGE^^}_TRAIN_FAILED" | tee -a "$STATUS"
  exit "$train_rc"
fi

CANDIDATE=$ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then
  echo "FAILED_INFRA_MISSING_${STAGE^^}_CHECKPOINT path=$CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

set +e
PYTHONUNBUFFERED=1 "$PY" "$REMOTE_ROOT/experience_docx/tools/eval_haze4k_checkpoint_compare.py" \
  --data_dir "$DATA" \
  --original_checkpoint "$A0" \
  --original_arch official_convir \
  --original_mode original \
  --original_name A0 \
  --candidate_checkpoint "$CANDIDATE" \
  --candidate_arch dta \
  --candidate_mode original \
  --candidate_name "DTA_${STAGE}_final" \
  --dta_depth_cache_dir "$DEPTH_CACHE" \
  --dta_eval_depth_split test \
  --candidate_dta_gate_bias -7.0 \
  --candidate_dta_gate_limit 0.03 \
  --output_dir "$COMPARE_DIR" \
  --tag "$TAG" \
  --max_images "$MAX_IMAGES" \
  2>&1 | tee "$EVAL_LOG"
eval_rc=${PIPESTATUS[0]}
set -e
echo "${STAGE}_eval_done rc=$eval_rc haze4k_dta_lowgate $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$eval_rc" -eq 0 ]]; then
  echo "DTA_LOWGATE_${STAGE^^}_TRAIN_EVAL_OK" | tee -a "$STATUS"
else
  echo "DTA_LOWGATE_${STAGE^^}_EVAL_FAILED" | tee -a "$STATUS"
fi
exit "$eval_rc"
