#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v32}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
EVAL_SPLIT=${EVAL_SPLIT:-fold0_val}
SOURCE_RUN_ID=${SOURCE_RUN_ID:-v31_wg18_base_s008_b14_seed3407_f0}
RUN_ID=${RUN_ID:-v32_ctdg_diag_wg18_base_s008_b14_seed3407_f0}
CANDIDATE_NAME=${CANDIDATE_NAME:-wg18_base_s008_b14}
CANDIDATE=${CANDIDATE:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-taillite/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-DAPC-DepthDirectTail-wg18_base_s008_b14-seed3407-f0-scout5full/Training-Results/Final.pkl}
MAX_IMAGES=${MAX_IMAGES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "dta_v3_2_ctdg_safemix_audit_start run_id=$RUN_ID source_run_id=$SOURCE_RUN_ID $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CANDIDATE"
  echo "candidate_name=$CANDIDATE_NAME"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_2_AUDIT_MISSING_CANDIDATE $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_v32_action_diagnostics.py \
  --run_id "$RUN_ID" \
  --candidate_name "$CANDIDATE_NAME" \
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
  --alpha_values 0.10,0.20,0.35,0.50,0.75,1.00 \
  --coverages 25,40,60,80,100 \
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
  --dta_phase depth \
  --dta_ablation full \
  2>&1 | tee "$EVID/dta_v3_2_action_diagnostics_${RUN_ID}.log"
diag_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_2_action_diagnostics_done rc=$diag_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$diag_rc" -ne 0 ]]; then exit "$diag_rc"; fi

for A_MODE in fallback gt; do
  true_csv=$(ls "$EVID"/dta_v3_1_${SOURCE_RUN_ID}_${A_MODE}_invert_compare/scout_eval_per_image_*.csv | head -n 1)
  zero_csv=$(ls "$EVID"/dta_v3_1_${SOURCE_RUN_ID}_${A_MODE}_zero_compare/scout_eval_per_image_*.csv | head -n 1)
  shuffle_csv=$(ls "$EVID"/dta_v3_1_${SOURCE_RUN_ID}_${A_MODE}_shuffle_compare/scout_eval_per_image_*.csv | head -n 1)
  normal_csv=$(ls "$EVID"/dta_v3_1_${SOURCE_RUN_ID}_${A_MODE}_normal_compare/scout_eval_per_image_*.csv | head -n 1)
  selected_csv="$EVID/per_image_delta_matrix_${SOURCE_RUN_ID}_${A_MODE}_risk_selected.csv"

  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_selector_metrics.py \
    --mode correction \
    --true_csv "$true_csv" \
    --zero_csv "$zero_csv" \
    --shuffle_csv "$shuffle_csv" \
    --normal_csv "$normal_csv" \
    --selected_csv "$selected_csv" \
    --output_json "$EVID/selector_metric_correction_report_${RUN_ID}_${A_MODE}.json" \
    2>&1 | tee "$EVID/selector_metric_correction_${RUN_ID}_${A_MODE}.log"
  corr_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_2_selector_metric_correction_done rc=$corr_rc run_id=$RUN_ID airlight=$A_MODE $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$corr_rc" -ne 0 ]]; then exit "$corr_rc"; fi

  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_selector_metrics.py \
    --mode nested \
    --true_csv "$true_csv" \
    --zero_csv "$zero_csv" \
    --shuffle_csv "$shuffle_csv" \
    --normal_csv "$normal_csv" \
    --output_json "$EVID/nested_selector_smoke_f0_${RUN_ID}_${A_MODE}.json" \
    --thresholds_csv "$EVID/nested_selector_smoke_f0_thresholds_${RUN_ID}_${A_MODE}.csv" \
    --risk_coverage_csv "$EVID/risk_coverage_curve_f0_nested_${RUN_ID}_${A_MODE}.csv" \
    --internal_folds 5 \
    --min_coverage 0.15 \
    --max_coverage 0.95 \
    2>&1 | tee "$EVID/nested_selector_smoke_f0_${RUN_ID}_${A_MODE}.log"
  nested_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_2_nested_selector_smoke_done rc=$nested_rc run_id=$RUN_ID airlight=$A_MODE $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$nested_rc" -ne 0 ]]; then exit "$nested_rc"; fi
done

echo "DTA_V3_2_CTDG_SAFEMIX_AUDITS_OK run_id=$RUN_ID" | tee -a "$STATUS"
