#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/workspace/ConvIR-B-v04e-repro-826caaf-bundle
OUT=$ROOT/experience_docx/experiment_logs/haze4k_rootcause_preexp_20260604/visual_v04e_formal
DATA=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
SEL=/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl
CORR_JSON=$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/correctability_traincalib_apdr_v0_4_correctability_traincalib_sigma3_seed3407.json
CORR_CSV=$ROOT/experience_docx/experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/correctability_traincalib_train_oof_apdr_v0_4_correctability_traincalib_sigma3_seed3407.csv
DEPTH=/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf
cd "$ROOT/Dehazing/ITS"
{
  echo "visual_v04e_formal_start $(date -Iseconds)"
  echo "root $ROOT"
  echo "out $OUT"
  echo "candidate_mappers corrected_kernel_names"
} | tee "$OUT/status.txt"
/root/miniconda3/envs/convir-cu128/bin/python "$ROOT/experience_docx/tools/audit_haze4k_apdr_v0_4_visual_regression_grids.py" \
  --mode v04e \
  --data_dir "$DATA" \
  --selector_checkpoint "$SEL" \
  --correctability_json "$CORR_JSON" \
  --correctability_train_csv "$CORR_CSV" \
  --depth_cache_dir "$DEPTH" \
  --output_dir "$OUT" \
  --tag visual_v04e_formal_sigma3_seed3407 \
  --basis_num_images 0 \
  --fold_count 5 \
  --seed 3407 \
  --device cuda \
  --pca_device cpu \
  --residual_max 0.04 \
  --kernel_size 31 \
  --sigma 3.0 \
  --low_size 32 \
  --k_values 16 \
  --candidate_k_values 16 \
  --candidate_mappers global_plus_spatial_kernel_knn_9,convir_spatial_kernel_knn_9,spatial_priors_kernel_knn_9,spatial_priors_ridge_10,global_mean_coeff \
  --scales 0.25,0.50,0.75,1.00 \
  --projection_ridge 1e-5 \
  --ridge_values 0.01,0.1,1.0,10.0 \
  --pls_components 4,8,16 \
  --knn_values 5,9 \
  --spatial_grid 4 \
  --spatial_proj_channels 8 \
  --max_per_group 12 \
  --progress_freq 100 \
  > "$OUT/visual_v04e_formal.log" 2>&1
rc=$?
echo "visual_v04e_formal_done rc=$rc $(date -Iseconds)" | tee -a "$OUT/status.txt"
exit $rc
