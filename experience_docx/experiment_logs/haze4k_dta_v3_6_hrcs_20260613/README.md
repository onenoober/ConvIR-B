# Haze4K DTA-v3.6 HRCS Evidence

Date: 2026-06-13

Status: `PLANNED_RELAXED_SELECTOR_FIRST_USER_LOCKED_TEST_OVERRIDE_PENDING`

Route card: `experience_docx/experiment_cards/2026-06-13-haze4k-dta-v3-6-hrcs.md`
Central index: `experience_docx/EXPERIMENT_INDEX.md`
Family summary: `experience_docx/family_summaries/dta_family_summary.md`

## Runtime Contract

- Host: `convir-4090`.
- Workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-6-hrcs`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`.
- v3.5 source evidence: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-v3-5-fdf-rcs-lite/experience_docx/experiment_logs/haze4k_dta_v3_5_fdf_rcs_lite_20260612/`.
- Locked test: blocked by default; user requested one later relaxed fixed-policy override.

## Primary Artifacts

- `status.txt`
- `run_dta_v3_6_hrcs_phase_a_convir4090.sh`
- `v36_high_coverage_rejection_curve.csv`
- `v36_high_coverage_rejection_curve_aggregate.csv`
- `v36_risk_feature_auc_report.csv`
- `v36_selector_reliability_bins.csv`
- `v36_selector_error_table.csv`
- `v36_action_bank_oracle_vs_selector.csv`
- `v36_selector_summary.json`
- `v36_selector_best_configs.csv`

## Initial Plan

Phase A/B are selector-only postprocesses over the existing v3.5 OOF table. They do not retrain ConvIR-B, do not touch locked test, and report both strict and relaxed gates.

Phase C will launch 5 folds x 3 seeds candidate evidence for L1/L2/L3 only if the fixed selector policy is worth validating. Phase D is a one-shot locked test only after the policy is sealed from train-derived evidence.

## Decision State

Pending cloud Phase A postprocess on `convir-4090`.
