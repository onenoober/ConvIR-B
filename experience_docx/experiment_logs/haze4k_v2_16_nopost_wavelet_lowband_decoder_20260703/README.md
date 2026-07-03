# Haze4K v2.16 NoPost Wavelet Lowband Decoder

Status: `PLANNED_T0_T1_ONLY`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-16-nopost-wavelet-lowband-decoder`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Base route:
`github/codex/haze4k-official-arch-anchor`

Previous v2.13 feature table input:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv`

Previous v2.15 evidence inputs:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-15-nopost-spatial-internal-risk-audit/experience_docx/experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703/`

Locked Haze4K test: untouched and blocked.

## Plan

Run T0/T1 only:

- target-decoupling audit between WD0375 severe-risk, A0 weakness, and
  lowband need;
- wavelet lowband headroom audit using train-derived A0 prediction and GT as
  oracle-only evidence;
- decide whether T2 contract/identity is allowed.

No training, checkpoint selection, locked Haze4K command, inference demo, or
learned RGB post-correction is part of this run.
