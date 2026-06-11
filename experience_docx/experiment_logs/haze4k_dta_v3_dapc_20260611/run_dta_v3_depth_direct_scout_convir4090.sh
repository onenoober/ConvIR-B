#!/usr/bin/env bash
set -euo pipefail
STAGE=${1:-scout5full}
TRAIN_DEPTH_MODE=${2:-invert}
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
  scout5full) NUM_EPOCH=5; STOP_EPOCH=5; SAVE_FREQ=1; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=1; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  scout10full) NUM_EPOCH=10; STOP_EPOCH=10; SAVE_FREQ=5; VALID_FREQ=${VALID_FREQ:-9999}; MOD_STATS_FREQ=2; MAX_IMAGES=${MAX_IMAGES:-0} ;;
  *) echo "Unsupported STAGE=$STAGE" >&2; exit 64 ;;
esac
case "$TRAIN_DEPTH_MODE" in
  invert|normal|zero|shuffle) ;;
  *) echo "Unsupported TRAIN_DEPTH_MODE=$TRAIN_DEPTH_MODE" >&2; exit 65 ;;
esac
RUN_ID=${STAGE}_depthDirect_${TRAIN_DEPTH_MODE}_seed${SEED}_f${FOLD}
MODEL_NAME=ConvIR-Haze4K-DTA-v3-DAPC-DepthDirect-${TRAIN_DEPTH_MODE}-seed${SEED}-f${FOLD}-${STAGE}
TRAIN_LOG=$EVID/dta_v3_${RUN_ID}_train.log
MATRIX_MANIFEST=$EVID/dta_v3_${RUN_ID}_matrix_manifest.json
CONTACT_DIR=$EVID/tail_regression_contact_sheet/$RUN_ID
mkdir -p "$EVID" "$CONTACT_DIR"
{
  echo "depth_direct_start run_id=$RUN_ID train_depth_mode=$TRAIN_DEPTH_MODE stage=$STAGE $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "train_split=$TRAIN_SPLIT"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "init=A0_partial train_scope=dta_depth_only r0_scale=0 gate_limit=0.12 gamma=0.20 beta=0.10"
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
  --dta_grad_clip_norm 0.03 \
  --init_model "$A0" \
  --init_model_partial \
  --partial_new_prefixes DTA. \
  --train_scope dta_depth_only \
  --dta_depth_cache_dir "$DEPTH" \
  --dta_train_depth_split train \
  --dta_eval_depth_split train \
  --dta_require_depth \
  --dta_depth_mode "$TRAIN_DEPTH_MODE" \
  --dta_phase depth \
  --dta_ablation full \
  --dta_prior_channels 32 \
  --dta_gate_bias -5.0 \
  --dta_gate_limit 0.12 \
  --dta_gamma_limit 0.20 \
  --dta_beta_limit 0.10 \
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
  --dta_ref_preserve_weight 0.03 \
  --dta_tail_guard_weight 0.03 \
  --dta_tail_guard_margin 0.0 \
  --dta_mask_budget_weight 0.001 \
  --split_json "$SPLIT_JSON" \
  --split_name "$TRAIN_SPLIT" \
  2>&1 | tee "$TRAIN_LOG"
train_rc=${PIPESTATUS[0]}
set -e
echo "depth_direct_train_done rc=$train_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$train_rc" -ne 0 ]]; then exit "$train_rc"; fi
CANDIDATE=$ITS/results/$MODEL_NAME/Training-Results/Final.pkl
if [[ ! -f "$CANDIDATE" ]]; then echo "MISSING_DEPTH_DIRECT_CHECKPOINT $CANDIDATE" | tee -a "$STATUS"; exit 3; fi
cd "$WORK"
printf '{"runs":[\n' > "$MATRIX_MANIFEST"
first=1
for EVAL_MODE in invert normal zero shuffle; do
  EVAL_RUN=${RUN_ID}_eval${EVAL_MODE}
  COMPARE_DIR=$EVID/dta_v3_${EVAL_RUN}_compare
  mkdir -p "$COMPARE_DIR"
  if [[ "$EVAL_MODE" == "shuffle" ]]; then
    "$PY" experience_docx/tools/audit_haze4k_dta_depth_pairing.py \
      --data_dir "$DATA" --depth_cache_dir "$DEPTH" --depth_split train --root_split train \
      --split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT" \
      --mode shuffle_eval_fixed_perm --offset 137 \
      --output_csv "$EVID/depth_eval_pairing_audit_${EVAL_RUN}.csv" \
      --output_json "$EVID/depth_eval_pairing_audit_${EVAL_RUN}.json" \
      > "$EVID/depth_eval_pairing_audit_${EVAL_RUN}.log" 2>&1
  fi
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/eval_haze4k_checkpoint_compare.py \
    --data_dir "$DATA" \
    --original_checkpoint "$A0" \
    --original_arch official_convir \
    --original_name A0 \
    --candidate_checkpoint "$CANDIDATE" \
    --candidate_arch dta_v3 \
    --candidate_name "DTA_v3_${EVAL_RUN}" \
    --dta_depth_cache_dir "$DEPTH" \
    --dta_eval_depth_split train \
    --candidate_dta_variant v3 \
    --candidate_dta_depth_mode "$EVAL_MODE" \
    --candidate_dta_phase depth \
    --candidate_dta_ablation full \
    --candidate_dta_prior_channels 32 \
    --candidate_dta_gate_bias -5.0 \
    --candidate_dta_gate_limit 0.12 \
    --candidate_dta_gamma_limit 0.20 \
    --candidate_dta_beta_limit 0.10 \
    --candidate_dta_confidence_floor 0.30 \
    --candidate_dta_r0_residual_scale 0.0 \
    --candidate_dta_depth_residual_scale 0.08 \
    --candidate_dta_depth_mask_easy_budget 0.04 \
    --candidate_dta_depth_mask_dense_budget 0.14 \
    --candidate_dta_depth_mask_density_thresh 0.35 \
    --candidate_dta_depth_mask_bias -4.0 \
    --candidate_dta_phys_t_min 0.10 \
    --depth_shuffle_offset 137 \
    --split_json "$SPLIT_JSON" \
    --split_name "$EVAL_SPLIT" \
    --eval_root_split train \
    --output_dir "$COMPARE_DIR" \
    --tag "$EVAL_RUN" \
    --max_images "$MAX_IMAGES" \
    2>&1 | tee "$EVID/dta_v3_${EVAL_RUN}_eval.log"
  eval_rc=${PIPESTATUS[0]}
  set -e
  echo "depth_direct_eval_done rc=$eval_rc run_id=$EVAL_RUN $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$eval_rc" -ne 0 ]]; then exit "$eval_rc"; fi
  if [[ "$first" -eq 0 ]]; then printf ',\n' >> "$MATRIX_MANIFEST"; fi
  first=0
  label="$EVAL_MODE"
  if [[ "$EVAL_MODE" == "$TRAIN_DEPTH_MODE" ]]; then label="true"; fi
  printf '  {"label":"%s","train_depth":"%s","eval_depth":"%s","compare_dir":"%s"}' "$label" "$TRAIN_DEPTH_MODE" "$EVAL_MODE" "$COMPARE_DIR" >> "$MATRIX_MANIFEST"
done
printf '\n]}\n' >> "$MATRIX_MANIFEST"
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/aggregate_haze4k_dta_v3_controls.py \
  --manifest "$MATRIX_MANIFEST" \
  --output_matrix_json "$EVID/train_eval_depth_matrix_${RUN_ID}.json" \
  --output_matrix_csv "$EVID/train_eval_depth_matrix_${RUN_ID}.csv" \
  --output_attribution_csv "$EVID/r0_vs_rdepth_attribution_${RUN_ID}.csv" \
  --baseline_label zero \
  --true_label true \
  2>&1 | tee "$EVID/dta_v3_${RUN_ID}_aggregate.log"
agg_rc=${PIPESTATUS[0]}
set -e
echo "depth_direct_aggregate_done rc=$agg_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$agg_rc" -ne 0 ]]; then exit "$agg_rc"; fi
TRUE_COMPARE_DIR=$EVID/dta_v3_${RUN_ID}_eval${TRAIN_DEPTH_MODE}_compare
TRUE_CSV=$(ls "$TRUE_COMPARE_DIR"/scout_eval_per_image_*.csv | head -n 1)
set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_contact_sheet.py \
  --data_dir "$DATA" --depth_cache_dir "$DEPTH" --depth_split train --root_split train \
  --split_json "$SPLIT_JSON" --split_name "$EVAL_SPLIT" \
  --per_image_csv "$TRUE_CSV" \
  --a0_checkpoint "$A0" --candidate_checkpoint "$CANDIDATE" --candidate_arch dta_v3 \
  --output_dir "$CONTACT_DIR" --tag "$RUN_ID" --count 12 \
  --dta_variant v3 --dta_depth_mode "$TRAIN_DEPTH_MODE" --dta_phase depth --dta_ablation full \
  --dta_prior_channels 32 --dta_gate_bias -5.0 --dta_gate_limit 0.12 --dta_gamma_limit 0.20 --dta_beta_limit 0.10 \
  --dta_confidence_floor 0.30 --dta_r0_residual_scale 0.0 --dta_depth_residual_scale 0.08 \
  --dta_depth_mask_easy_budget 0.04 --dta_depth_mask_dense_budget 0.14 --dta_depth_mask_bias -4.0 \
  2>&1 | tee "$EVID/dta_v3_${RUN_ID}_contact_sheet.log"
contact_rc=${PIPESTATUS[0]}
set -e
echo "depth_direct_contact_sheet_done rc=$contact_rc run_id=$RUN_ID output=$CONTACT_DIR $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$contact_rc" -ne 0 ]]; then exit "$contact_rc"; fi
echo "DTA_V3_DEPTH_DIRECT_SCOUT_OK run_id=$RUN_ID checkpoint=$CANDIDATE matrix=$EVID/train_eval_depth_matrix_${RUN_ID}.json contact_sheet=$CONTACT_DIR" | tee -a "$STATUS"
