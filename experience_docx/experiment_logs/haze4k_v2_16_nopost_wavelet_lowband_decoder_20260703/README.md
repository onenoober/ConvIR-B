# Haze4K v2.16 NoPost Wavelet Lowband Decoder

Status: `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`

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

## WLDB-A Screen

Cloud run: `convir-4090`, `2026-07-03T16:22:46+08:00` to
`2026-07-03T16:34:35+08:00`.

Source commit: `0198474`.

WLDB-A trained seed `3407` for `20` epochs on train-derived fold split:

- train fold count: `1920`;
- validation fold0 count: `480`;
- trainable params: `2128`;
- frozen official params: `8630665`;
- locked Haze4K test: untouched.

Decision: `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`.

No evaluated checkpoint passed the screen gate. Best by mean dPSNR was
`model_5`, with:

- mean dPSNR: `+0.081889`;
- hard bottom25 dPSNR: `+0.105887`;
- easy top25 dPSNR: `+0.020994`;
- positive ratio: `0.662500`;
- severe loss count: `67/480`;
- strong-reference regressions: `48/120`.

`model_5` passed mean/hard/easy/positive/strong-reference checks but failed the
tail-safety gate (`severe_loss_count <= 12`). Later checkpoints reduced severe
loss but also lost mean and hard gains, and still failed severe-loss limits.

Conclusion: the lowband objective has real movement, but the current WLDB-A
training form is not tail-safe. Per protocol, stop this screen without
multi-seed expansion, longer training, locked-test use, or checkpoint
promotion.
