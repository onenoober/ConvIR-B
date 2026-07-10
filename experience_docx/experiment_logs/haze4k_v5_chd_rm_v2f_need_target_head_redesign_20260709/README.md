# CHD-RM v2f Need Target/Head Redesign Evidence

Status: `PAUSED_AFTER_F4B_GATE_FAIL`

Route card: `experience_docx/experiment_cards/haze4k-chd-rm-v2f-need-target-head-redesign.md`

Central index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## First-Stage Results

F0-F3/F2 completed on `convir-4090` with locked Haze4K test usage `none`.

Primary first-stage files:

- `v2f_source_of_truth_manifest.json`
- `v2e_d7c_candidate_reproduction.csv`
- `v2e_d7c_rp_reproduction.csv`
- `ldhn_target_autopsy_summary.json`
- `target_variant_density_proxy_correlation.csv`
- `feature_probe_ldhn_vs_ldln_summary.csv`
- `v2f_first_stage_closeout.json`

Key first-stage observations:

- LDHN pixel coverage: `0.08988972981770833`.
- LDHN core fraction of LDHN: `0.569798970635499`.
- LDHN unstable fraction of LDHN: `0.04701398288013833`.
- Best frozen feature probe: `feature_set_2` + `mlp`, AUROC
  `0.8107264347671554`, AUPRC `0.807792756659645`.
- Density-conditioned target removes the global density proxy signal:
  density Spearman `0.007215705298292346` vs global `0.31464418569286756`.

## Current Decision

F4 density-stratified frozen-side head canary completed with
`COMPLETED_GATE_FAIL`, and the supplemental F4b tail-rescue matrix also
completed with `COMPLETED_GATE_FAIL`. This was not a v3, D2, or RARM step.
ConvIR-B A0 and D3 density stayed frozen, and the original v2e global
LDHN/false-tail gate remained the primary decision contract.

Primary F4 files:

- `f4_authorization_record.md`
- `run_v2f_stratified_head_canary.sh`
- `stratified_head_ablation_summary.csv`
- `stratified_head_threshold_curve.csv`
- `stratified_head_per_image_safety_metrics.csv`
- `v2f_f4_stratified_head_closeout.json`
- `v2f_f4_stratified_head_summary.md`

Primary F4b files:

- `../haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/README.md`
- `../haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/v2f_f4b_tail_rescue_closeout.json`
- `../haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/v2f_f4b_tail_rescue_summary.md`
- `../haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/f4b_tail_rescue_matrix_summary.csv`

Key closeout: F4 selected variants had no safe+LDHN point, and F4b selected
variants also had `safe_and_ldhn_points = 0`. Best F4b safe LDHN recall was
only `0.0523`, while high-LDHN-recall variants had false-p95 near `0.9895` to
`1.0000`.

Forbidden until later written authorization: F5 controls, D2, ConvIR-B unfreeze,
v3, RARM connection/training, and locked Haze4K test. Do not repeat F4/F4b
strength sweeps without changing target semantics or available information.
