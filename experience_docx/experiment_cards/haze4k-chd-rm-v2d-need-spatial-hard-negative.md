# Haze4K CHD-RM v2d Need Spatial Hard-Negative

Status: `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3`

Evidence root:

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/`

Runtime source:

- Host: `convir-4090`
- Branch: `codex/haze4k-v5-v2d-chd-rm-need-spatial-hard-negative`
- Base: `codex/haze4k-v5-v2c-chd-rm-need-coverage-calibration`
- Locked Haze4K test usage: none

Route identity:

v2d continues CHD-RM after v2c scale calibration failed. It keeps ConvIR-B frozen, does not change A0, and does not connect or train RARM.

Main result:

D7a/D7b shallow-head hard-negative repair is insufficient. D7c frozen multi-context top-k hard-negative head is the best candidate:

- Spearman `0.5175`
- AUROC high-vs-low `0.8456`
- AUPRC high `0.6442`
- selected train_inner threshold `0.5773`
- coverage `0.3027`
- false-strong global `0.0030`
- false-strong p90 `0.0246`
- monotonic `4/4`

Pause reason:

Controls are not clean enough for v3. Shuffled/random controls fail the full R_need gate but retain weak density/context proxy signal, so v3 no-op RARM remains blocked.

Next allowed work:

- D7c stricter fixed-permutation control.
- Density-only matched-threshold control.
- Low-density high-need recall protection audit.

Forbidden:

- D2.
- RARM connection/training.
- v3 expansion or no-op audit before controls are clean.
- Locked Haze4K test.
