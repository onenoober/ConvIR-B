#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/home/caozhiyang/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune}
PY=${PY:-$BASE/envs/convir-cu128/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=$EVID/dta_v3_haze4k_oof_splits_seed3407.json
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$EVID"
{
  echo "preflight_start dta_v3 $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "depth=$DEPTH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} | tee -a "$STATUS"
cd "$WORK"
{
  git branch --show-current
  git rev-parse --short HEAD
  git status --short
} | tee -a "$STATUS"

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/make_haze4k_dta_oof_splits.py \
  --data_dir "$DATA" \
  --output "$SPLIT_JSON" \
  --folds 5 \
  --seed 3407 \
  > "$EVID/dta_v3_oof_splits.log" 2>&1
split_rc=$?
set -e
echo "oof_splits_done rc=$split_rc dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$split_rc" -ne 0 ]]; then exit "$split_rc"; fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/check_haze4k_dta_preflight.py \
  --checkpoint "$A0" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split train \
  --output_json "$EVID/dta_v3_preflight_r0.json" \
  --arch dta_v3 \
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
  --use_trans_gt \
  --trans_weight 0.02 \
  --phys_weight 0.005 \
  --noop_tolerance 0.0000001 \
  > "$EVID/dta_v3_preflight_r0.log" 2>&1
r0_rc=$?
set -e
echo "preflight_r0_done rc=$r0_rc dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$r0_rc" -ne 0 ]]; then exit "$r0_rc"; fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/check_haze4k_dta_preflight.py \
  --checkpoint "$A0" \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split train \
  --output_json "$EVID/dta_v3_preflight_depth_bounded.json" \
  --arch dta_v3 \
  --dta_variant v3 \
  --dta_depth_mode invert \
  --dta_phase depth \
  --dta_ablation full \
  --dta_prior_channels 32 \
  --dta_gate_bias -5.0 \
  --dta_gate_limit 0.10 \
  --dta_gamma_limit 0.16 \
  --dta_beta_limit 0.08 \
  --dta_confidence_floor 0.30 \
  --dta_depth_mask_easy_budget 0.04 \
  --dta_depth_mask_dense_budget 0.12 \
  --dta_depth_mask_bias -4.0 \
  --use_trans_gt \
  --trans_weight 0.02 \
  --phys_weight 0.005 \
  --noop_tolerance 0.02 \
  > "$EVID/dta_v3_preflight_depth_bounded.log" 2>&1
depth_rc=$?
set -e
echo "preflight_depth_done rc=$depth_rc dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$depth_rc" -ne 0 ]]; then exit "$depth_rc"; fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_depth_pairing.py \
  --data_dir "$DATA" \
  --depth_cache_dir "$DEPTH" \
  --depth_split train \
  --root_split train \
  --split_json "$SPLIT_JSON" \
  --split_name fold0_val \
  --mode shuffle_eval_fixed_perm \
  --offset 137 \
  --output_csv "$EVID/depth_eval_pairing_audit.csv" \
  --output_json "$EVID/depth_eval_pairing_audit.json" \
  > "$EVID/depth_eval_pairing_audit.log" 2>&1
pair_rc=$?
set -e
echo "pairing_audit_done rc=$pair_rc dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$pair_rc" -ne 0 ]]; then exit "$pair_rc"; fi

echo "preflight_done rc=0 dta_v3 $(date --iso-8601=seconds)" | tee -a "$STATUS"
echo "DTA_V3_PREFLIGHT_OK" | tee -a "$STATUS"
