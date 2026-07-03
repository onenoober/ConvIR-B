# Haze4K v2.15 NoPost Spatial/Internal Risk Audit

Status: `PLANNED_N1S_ONLY_LOCKED_TEST_UNTOUCHED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-15-nopost-spatial-internal-risk-audit`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Previous v2.13 feature table:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv`

Previous v2.14 predictions:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-14-nopost-runtime-evidence-audit/experience_docx/experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/v214_n1r_oof_predictions.csv`

Locked Haze4K test: untouched and blocked.

## Plan

Run S0-S4 only:

- lock the protocol and forbidden-feature contract;
- decompose v2.14 top-100 severe-risk failure;
- extract NoPost dense-map spatial features;
- extract internal feature-space sensitivity features;
- run 5-fold x 3-seed OOF ranking probes against the v2.14 hazy-runtime
  baseline.

No training, locked-test command, checkpoint write, or inference demo is part
of this route.
