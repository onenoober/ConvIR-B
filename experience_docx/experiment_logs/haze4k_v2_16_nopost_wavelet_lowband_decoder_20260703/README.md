# Haze4K v2.16 NoPost Wavelet Lowband Decoder

Status: `COMPLETED_T2_PASS_TRAINING_BLOCKED_PENDING_REVIEW`

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

## Closeout

Cloud run: `convir-4090`, `2026-07-03T15:18:01+08:00` to
`2026-07-03T15:25:30+08:00`.

Source commits:

- T0/T1 scaffold: `ba3c10c`;
- T2 contract/identity: `344b0df`.

T0/T1/T2 completed successfully. Locked Haze4K test remained untouched and no
training was launched.

T0 decision: `T0_TARGET_DECOUPLED`.

- WD0375 severe vs lowband-need Jaccard: `0.027917`;
- WD0375 severe vs A0 hard-bottom25 Jaccard: `0.000000`.

T1 decision: `T1_LOWBAND_HEADROOM_PASS_ALLOW_T2`.

- rows: `2400`;
- all-image LL oracle mean dPSNR: `14.998694`;
- A0 hard-bottom25 LL oracle mean dPSNR: `18.939359`;
- A0 easy-top25 LL oracle mean dPSNR: `11.853745`;
- severe LL-oracle regressions: `0`;
- lowband-need rate: `1.000000`.

T2 decision:
`T2_CONTRACT_IDENTITY_PASS_TRAINING_STILL_BLOCKED_PENDING_REVIEW`.

- forbidden symbol hits: `0`;
- identity samples: `32`;
- identity max abs diff: `1.7881393432617188e-07`;
- identity threshold: `1e-05`;
- official checkpoint keys loaded: `602`;
- new WLDB module keys missing by design: `6`.

Conclusion: v2.16 passes the no-training feasibility diagnostics and contract
identity gate. The next action is review before any WLDB-A training launch; T2
does not by itself authorize training or locked-test use.
