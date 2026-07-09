# Haze4K CHD-RM v2f Need Target/Head Redesign

Status: `PLANNED`

Evidence root:

`experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/`

Runtime source:

- Host: `convir-4090`
- Branch: `codex/haze4k-v5-v2f-chd-rm-need-target-head-redesign`
- Parent: `github/codex/haze4k-v5-v2e-chd-rm-d7c-control-recall-audit`
- Parent commit: `204897af07cb855a0af2f037edbd8cf42098bfeb`
- Route identity: v2-family frozen-side diagnostic continuation, not v3/RARM.
- Locked Haze4K test usage: none

## Objective

Diagnose and redesign the frozen-side `R_need` target/head so low-density
high-need recall and false-tail safety can pass together before any v3/RARM
work is considered.

## Fact Sources

- GitHub `main`: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.
- GitHub `main`: v2e evidence root
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709/`.
- Cloud runtime: `convir-4090`
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2e-chd-rm-d7c-control-recall-audit`.

## Route Framing

v2f continues the v2e diagnosis because v2e proved D7c has real `R_need`
signal but no safe LDHN recall-protected operating point. v2f does not promote
D7c to RARM and does not run restoration evaluation. It tests target stability,
within-density need separability, and frozen feature separability.

## Forbidden Flow

- D2: not allowed.
- ConvIR-B unfreeze: not allowed.
- RARM connection/training: not allowed.
- v3 no-op RARM audit: not allowed from the v2e failed RP state.
- Haze4K locked test: not allowed.
- Further D7c-RP strength sweeping as the next step: not allowed.

## First Stage

The first authorized stage is diagnostic only:

- F0: reproduce and freeze v2e source-of-truth metrics.
- F1: LDHN target autopsy and stability classification.
- F2-lite: frozen feature LDHN-vs-LDLN separability probes.
- F3: density-conditioned and excess-over-density target transforms.

F4 density-stratified head canary is not authorized until F1/F2/F3 evidence is
written and supports it.

## Metric Contract

All target transforms are fitted on `train_inner` and evaluated on `val_inner`.
The baseline v2e gate remains:

- Spearman >= 0.50, AUROC >= 0.83, AUPRC >= 0.62.
- false_global <= 0.01, false_p90 <= 0.05, false_p95 <= 0.10.
- LDHN recall >= 0.10, preferred >= 0.12.
- Density independence must beat density-only matched controls.
- Controls must remain clean before any RARM-adjacent step.

## Resource Preflight

- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Split JSON:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2c-chd-rm-need-coverage-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`.
- D3 density artifact:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt`.
- D7c top-k artifact:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt`.
- D7c HN artifact:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_hn_ordinal_head.pt`.

