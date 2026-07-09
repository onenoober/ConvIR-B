# CHD-RM v2f Need Target/Head Redesign Evidence

Status: `PAUSE_V2F_F4B_NO_SAFE_LDHN_POINT_NO_F5_NO_V3`

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

## F4/F4b Decision

F4 density-stratified frozen-side head canary completed on `convir-4090` with
status `COMPLETED_GATE_FAIL`. This was not a v3, D2, or RARM step. ConvIR-B A0
and D3 density remained frozen, and the original v2e global LDHN/false-tail
gate remained the primary decision contract.

Primary F4 files:

- `f4_authorization_record.md`
- `run_v2f_stratified_head_canary.sh`
- `stratified_head_ablation_summary.csv`
- `stratified_head_threshold_curve.csv`
- `stratified_head_per_image_safety_metrics.csv`
- `v2f_f4_stratified_head_closeout.json`
- `v2f_f4_stratified_head_summary.md`

F4 found no selected variant with a safe LDHN operating point. The supplemental
F4b tail-rescue matrix then tested whether additional tail pressure could
rescue the route. F4b also completed with status `COMPLETED_GATE_FAIL`.

Supplemental F4b evidence root:

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_f4b_tail_rescue_20260709/`

Primary F4b files:

- `README.md`
- `f4b_tail_rescue_matrix_summary.csv`
- `v2f_f4b_tail_rescue_closeout.json`
- `v2f_f4b_tail_rescue_summary.md`

F4b closeout:

- `selected_gate_pass_any_variant=false`
- `safe_and_ldhn_point_any_variant=false`
- best safe LDHN recall `0.05233281880197182`
- best selected LDHN recall `0.6660676374862502`, but with unsafe false-tail
- minimum first LDHN-passing false-p95 `0.2`

Decision: keep v2f paused. Do not run F5, ConvIR-B unfreeze, D2, v3, RARM
connection/training, or locked Haze4K test from this evidence state.
