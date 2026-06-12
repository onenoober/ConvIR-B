# Haze4K DTA-v3.5 FDF-RCS-Lite Evidence

Date: 2026-06-12

Status: `PLANNED_RELAXED_TRAIN_DERIVED_FLOW`

Route card: `experience_docx/experiment_cards/2026-06-12-haze4k-dta-v3-5-fdf-rcs-lite.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- Locked test: blocked; train-derived OOF and nested calibration only.

## Planned Artifacts

- `status.txt`
- `run_dta_v3_5_fdf_rcs_lite_convir4090.sh`
- `launch_dta_v3_5_fdf_rcs_lite_triage_convir4090.sh`
- `dta_v3_5_fdf_rcs_triage_summary.json/csv`
- `dta_v3_5_fdf_rcs_triage_variant_summary.csv`
- `v35_oof_per_image_action_table.csv`
- `v35_oracle_risk_coverage_curve.csv`
- `v35_selector_nested_calibration_report.json/csv`
- `v35_selector_nested_selected_images.csv`

## Current State

Implementation and launch scripts are staged for cloud sync. No cloud runtime has been launched from this route yet.
