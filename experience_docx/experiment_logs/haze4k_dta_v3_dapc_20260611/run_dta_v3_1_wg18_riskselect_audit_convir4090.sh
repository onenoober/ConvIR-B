#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${RUN_ID:-v31_wg18_base_s008_b14_seed3407_f0}
BASE=${BASE:-/sda/home/wangyuxin/ConvIR-B}
WORK=${WORK:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-v31}
PY=${PY:-$BASE/envs/convir-cu121/bin/python}
DATA=${DATA:-$BASE/datasets/Haze4K/Haze4K}
A0=${A0:-$BASE/checkpoints/official/Haze4K/haze4k-base.pkl}
DEPTH=${DEPTH:-$BASE/depth_cache/depth_anything_v2_small_hf}
CANDIDATE=${CANDIDATE:-$BASE/repos/ConvIR-B-dta-v3-dapc-finetune-taillite/Dehazing/ITS/results/ConvIR-Haze4K-DTA-v3-DAPC-DepthDirectTail-wg18_base_s008_b14-seed3407-f0-scout5full/Training-Results/Final.pkl}
EVID=$WORK/experience_docx/experiment_logs/haze4k_dta_v3_dapc_20260611
STATUS=$EVID/status.txt
SPLIT_JSON=${SPLIT_JSON:-$EVID/dta_v3_haze4k_oof_splits_seed3407.json}
EVAL_SPLIT=${EVAL_SPLIT:-fold0_val}
MAX_IMAGES=${MAX_IMAGES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$EVID"
{
  echo "dta_v3_1_audit_start run_id=$RUN_ID $(date --iso-8601=seconds)"
  echo "work=$WORK"
  echo "python=$PY"
  echo "data=$DATA"
  echo "a0=$A0"
  echo "candidate=$CANDIDATE"
  echo "depth=$DEPTH"
  echo "split_json=$SPLIT_JSON"
  echo "eval_split=$EVAL_SPLIT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "locked_test_touched=false"
} | tee -a "$STATUS"

cd "$WORK"
{ git branch --show-current; git rev-parse --short HEAD; git status --short; } | tee -a "$STATUS"
if [[ ! -f "$CANDIDATE" ]]; then
  echo "DTA_V3_1_AUDIT_MISSING_CANDIDATE $CANDIDATE" | tee -a "$STATUS"
  exit 3
fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_output_semantics.py \
  --data_dir "$DATA" \
  --checkpoint "$A0" \
  --depth_cache_dir "$DEPTH" \
  --output_json "$EVID/output_semantics_audit.json" \
  --split_json "$SPLIT_JSON" \
  --split_name "$EVAL_SPLIT" \
  --eval_root_split train \
  --depth_split train \
  --max_images 8 \
  2>&1 | tee "$EVID/output_semantics_audit.log"
sem_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_1_output_semantics_audit_done rc=$sem_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$sem_rc" -ne 0 ]]; then exit "$sem_rc"; fi

set +e
PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/audit_haze4k_dta_airlight_gap.py \
  --data_dir "$DATA" \
  --original_checkpoint "$A0" \
  --candidate_checkpoint "$CANDIDATE" \
  --depth_cache_dir "$DEPTH" \
  --output_csv "$EVID/airlight_train_eval_gap.csv" \
  --output_summary_json "$EVID/airlight_oracle_vs_pred_summary.json" \
  --split_json "$SPLIT_JSON" \
  --split_name "$EVAL_SPLIT" \
  --eval_root_split train \
  --depth_split train \
  --max_images "$MAX_IMAGES" \
  --dta_depth_mode invert \
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
  --dta_depth_mask_bias -4.0 \
  --dta_phase depth \
  --dta_ablation full \
  2>&1 | tee "$EVID/airlight_train_eval_gap.log"
air_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_1_airlight_gap_audit_done rc=$air_rc run_id=$RUN_ID $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$air_rc" -ne 0 ]]; then exit "$air_rc"; fi

for A_MODE in fallback gt; do
  MANIFEST=$EVID/dta_v3_1_${RUN_ID}_${A_MODE}_matrix_manifest.json
  printf '{"runs":[\n' > "$MANIFEST"
  first=1
  for EVAL_MODE in invert zero shuffle normal; do
    EVAL_RUN=${RUN_ID}_${A_MODE}_${EVAL_MODE}
    COMPARE_DIR=$EVID/dta_v3_1_${EVAL_RUN}_compare
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
      --candidate_name "DTA_v31_${EVAL_RUN}" \
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
      --depth_shuffle_offset 137 \
      --split_json "$SPLIT_JSON" \
      --split_name "$EVAL_SPLIT" \
      --eval_root_split train \
      --output_dir "$COMPARE_DIR" \
      --tag "$EVAL_RUN" \
      --max_images "$MAX_IMAGES" \
      2>&1 | tee "$EVID/dta_v3_1_${EVAL_RUN}_eval.log"
    eval_rc=${PIPESTATUS[0]}
    set -e
    echo "dta_v3_1_eval_done rc=$eval_rc run_id=$EVAL_RUN $(date --iso-8601=seconds)" | tee -a "$STATUS"
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
    2>&1 | tee "$EVID/dta_v3_1_${RUN_ID}_${A_MODE}_aggregate.log"
  agg_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_1_aggregate_done rc=$agg_rc run_id=$RUN_ID airlight=$A_MODE $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$agg_rc" -ne 0 ]]; then exit "$agg_rc"; fi

  true_csv=$(ls "$EVID"/dta_v3_1_${RUN_ID}_${A_MODE}_invert_compare/scout_eval_per_image_*.csv | head -n 1)
  zero_csv=$(ls "$EVID"/dta_v3_1_${RUN_ID}_${A_MODE}_zero_compare/scout_eval_per_image_*.csv | head -n 1)
  shuffle_csv=$(ls "$EVID"/dta_v3_1_${RUN_ID}_${A_MODE}_shuffle_compare/scout_eval_per_image_*.csv | head -n 1)
  normal_csv=$(ls "$EVID"/dta_v3_1_${RUN_ID}_${A_MODE}_normal_compare/scout_eval_per_image_*.csv | head -n 1)
  set +e
  PYTHONUNBUFFERED=1 "$PY" experience_docx/tools/select_haze4k_dta_risk_oof.py \
    --true_csv "$true_csv" \
    --zero_csv "$zero_csv" \
    --shuffle_csv "$shuffle_csv" \
    --normal_csv "$normal_csv" \
    --output_json "$EVID/risk_selector_oof_calibration_${RUN_ID}_${A_MODE}.json" \
    --threshold_trace_csv "$EVID/risk_selector_threshold_trace_${RUN_ID}_${A_MODE}.csv" \
    --selected_csv "$EVID/per_image_delta_matrix_${RUN_ID}_${A_MODE}_risk_selected.csv" \
    2>&1 | tee "$EVID/risk_selector_${RUN_ID}_${A_MODE}.log"
  selector_rc=${PIPESTATUS[0]}
  set -e
  echo "dta_v3_1_risk_selector_done rc=$selector_rc run_id=$RUN_ID airlight=$A_MODE $(date --iso-8601=seconds)" | tee -a "$STATUS"
  if [[ "$selector_rc" -ne 0 ]]; then exit "$selector_rc"; fi
done

CONTACT_DIR=$EVID/tail_regression_contact_sheet/${RUN_ID}_fallback
mkdir -p "$CONTACT_DIR"
true_csv=$(ls "$EVID"/dta_v3_1_${RUN_ID}_fallback_invert_compare/scout_eval_per_image_*.csv | head -n 1)
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
  2>&1 | tee "$EVID/dta_v3_1_${RUN_ID}_contact_sheet.log"
contact_rc=${PIPESTATUS[0]}
set -e
echo "dta_v3_1_contact_sheet_done rc=$contact_rc run_id=$RUN_ID output=$CONTACT_DIR $(date --iso-8601=seconds)" | tee -a "$STATUS"
if [[ "$contact_rc" -ne 0 ]]; then exit "$contact_rc"; fi

echo "DTA_V3_1_WG18_RISKSELECT_AUDIT_OK run_id=$RUN_ID" | tee -a "$STATUS"
