# CHD-RM v2d Need Spatial Hard-Negative Evidence

Status: `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3`

Final source of truth:

1. `v2d_overall_result_summary.md`
2. `v2d_overall_run_summary.json`
3. `experience_docx/experiment_cards/haze4k-chd-rm-v2d-need-spatial-hard-negative.md`

Older shallow-head source:

- `v2d_run_summary.json`
- `v2d_result_summary.md`

Final result:

D7c frozen multi-context top-k hard-negative is the strongest v2d candidate, with Spearman `0.5175`, AUROC `0.8456`, AUPRC `0.6442`, coverage `0.3027`, false-global `0.0030`, false-p90 `0.0246`, and false-p95 `0.0476`. Controls remained weak enough to block v3/RARM.

This route keeps ConvIR-B frozen, does not connect RARM, and does not use the locked Haze4K test.
