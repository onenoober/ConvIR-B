# CHD-RM v2f Need Target/Head Redesign Evidence

Status: `F4_AUTHORIZED_PENDING_CLOUD`

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

F4 density-stratified frozen-side head canary is authorized. This is not a v3,
D2, or RARM step. ConvIR-B A0 and D3 density remain frozen, and the original v2e
global LDHN/false-tail gate remains the primary decision contract.

Primary F4 files:

- `f4_authorization_record.md`
- `run_v2f_stratified_head_canary.sh`
- `stratified_head_ablation_summary.csv`
- `stratified_head_threshold_curve.csv`
- `stratified_head_per_image_safety_metrics.csv`
- `v2f_f4_stratified_head_closeout.json`

Forbidden until later written authorization: D2, ConvIR-B unfreeze, v3, RARM
connection/training, and locked Haze4K test.
