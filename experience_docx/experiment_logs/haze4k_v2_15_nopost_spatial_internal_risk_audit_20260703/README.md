# Haze4K v2.15 NoPost Spatial/Internal Risk Audit

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_UNTOUCHED`

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

## Closeout

Cloud run: `convir-4090`, `2026-07-03T11:48:33+08:00` to
`2026-07-03T11:59:01+08:00`.

Source commit: `c11087f`.

Decision: `N1S_PARTIAL_INTERNAL_SIGNAL_NO_TRAINING`.

S2/S3 completed with `2400` rows, `13` dense maps, `1092` spatial feature
columns, and zero NaN/Inf values. Locked Haze4K test remained untouched and no
training was launched.

Primary S4 result at `WD0375_dPSNR <= -0.2`:

- hazy-runtime PR-AUC/top100 severe count: `0.149621` / `19`;
- best candidate `B5_internal_sensitivity` PR-AUC/top100 severe count:
  `0.132535` / `13`;
- PR-AUC delta: `-0.017086`;
- top100 enrichment delta: `-2.149254`;
- stable fold-seed units: `0/15`.

S1 decomposition showed the v2.14 all-runtime top100 had only `47` images in
common with hazy-runtime, lost `8` severe cases, and gained `49` false
positives.

Conclusion: spatial dense-map and internal sensitivity features do not fix the
severe-risk top-tail ranking bottleneck. N3/N4 remain blocked.

Raw S2/S3 feature tables remain cloud-only:

- `v215_s2_spatial_feature_rows.csv`
- `v215_s3_fam_response_features.csv`
- `v215_s3_skip_merge_disagreement.csv`
- `v215_s3_feature_jitter_consistency.csv`
