#!/usr/bin/env bash
set -euo pipefail
RUN_ID=${1:?run_id required}
MODEL_NAME=${2:?model_name required}
R0_SCALE=${3:-0.04}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
EVAL_SPLIT=${EVAL_SPLIT:-fold0_val}
STATUS=$EVID/status.txt
CANDIDATE=$WORK/Dehazing/ITS/results/$MODEL_NAME/Training-Results/Final.pkl
COMPARE_DIR=$EVID/dta_v3_${RUN_ID}_compare
PER_IMAGE=$COMPARE_DIR/scout_eval_per_image_${RUN_ID}.csv
CONTACT_DIR=$EVID/tail_regression_contact_sheet/$RUN_ID
CONTACT_LOG=$EVID/dta_v3_${RUN_ID}_contact_sheet.log
mkdir -p "$CONTACT_DIR"
if [[ ! -f "$CANDIDATE" ]]; then echo "MISSING_CONTACT_CANDIDATE $CANDIDATE" | tee -a "$STATUS"; exit 3; fi
if [[ ! -f "$PER_IMAGE" ]]; then echo "MISSING_CONTACT_PER_IMAGE $PER_IMAGE" | tee -a "$STATUS"; exit 4; fi
cd "$WORK"
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
  --count "${COUNT:-12}" \
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
echo "DTA_V3_CONTACT_SHEET_COMPLETE run_id=$RUN_ID output=$CONTACT_DIR" | tee -a "$STATUS"
