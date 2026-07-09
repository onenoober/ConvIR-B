# CHD-RM v2d Source Of Truth

Final source of truth:

1. `v2d_overall_result_summary.md`
2. `v2d_overall_run_summary.json`
3. `experience_docx/experiment_cards/haze4k-chd-rm-v2d-need-spatial-hard-negative.md`

Older shallow-head source:

- `v2d_run_summary.json`
- `v2d_result_summary.md`

Final decision:

`PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3`

Reason:

D7c frozen multi-context top-k hard-negative is the best v2d candidate, but shuffled/random/density controls remained too strong to authorize v3 or RARM.
