# Haze4K Root-Cause Pre-Experiments

Date: 2026-06-04

Status: root-cause evidence archived text-only; used by the DPGA-Lite v1.0
route card.

## Pointers

- Route card:
  `experience_docx/experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/dpga_family_summary.md`

## Primary Files

- `scout_eval_compare_a1_stop5_vs_a0.json`
- `scout_eval_compare_a1_best_vs_a0.json`
- `scout_eval_compare_a1_final_vs_a0.json`
- `scout_eval_bucket_analysis_a1_*_vs_a0.json`
- `scout_eval_per_image_a1_*_vs_a0.csv`
- `prior_predictability_full_rerun_after_reconnect/prior_predictability_summary_sigma3.json`
- `prior_predictability_full_rerun_after_reconnect/prior_predictability_coeff_cv_sigma3.csv`
- `prior_predictability_full_rerun_after_reconnect/prior_predictability_feature_rows_sigma3.csv`
- `visual_v04d_formal_rerun_after_reconnect/v04d_visual_regression_grid_summary_sigma3.json`
- `visual_v04e_formal_rerun_after_reconnect/v04e_visual_regression_grid_summary_sigma3.json`
- command logs and status files under the corresponding subdirectories.

## Key Result

The root-cause package supports the DPGA/v1.0 pivot away from output-end RGB
residual coefficient mapping:

- simple A1 finetuning did not stably improve A0 under this setup;
- depth, physics, and frequency priors carry deployable signal, but did not make
  output residual coefficients a safe action surface;
- severe/strong failures concentrated around sky, white-wall/white-ground,
  water/glass/airlight gradients, bright low-contrast regions, and strong-anchor
  cases where broad low-frequency/color shifts hurt preservation.

Decision label:

```text
ROOTCAUSE_SUPPORTS_IN_NETWORK_PRIOR_ROUTE_NOT_OUTPUT_RESIDUAL_MAPPING
```

## Artifact Boundary

This directory is synchronized as text-only evidence. Rendered visual grids,
sample images, checkpoints, datasets, tensor caches, arrays, and raw inference
outputs are intentionally excluded.
