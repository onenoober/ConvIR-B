# Haze4K v2.15 NoPost Spatial/Internal Risk Audit

Date: 2026-07-03

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_UNTOUCHED`

Branch: `codex/haze4k-v2-15-nopost-spatial-internal-risk-audit`

Base route: `github/codex/haze4k-v2-14-nopost-runtime-evidence-audit`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703/`

## Reason

v2.14 showed that the NoPost route is not blocked by contract, zero-init, or
ordinary ROC-AUC. The remaining blocker is severe-risk top-tail ranking:
`all_runtime` severe-risk PR-AUC and top-100 enrichment are below the
`hazy_runtime` baseline.

v2.15 tests whether the missing severe-risk signal is present in spatial
dense-map evidence or feature-space sensitivity evidence before any adapter
training is allowed.

## Scope

N1S is diagnostic-only:

- S0 protocol lock;
- S1 v2.14 top-100 failure decomposition;
- S2 NoPost dense-map spatial feature table;
- S3 internal feature-space sensitivity;
- S4 5-fold x 3-seed OOF ranking probe.

No N3/N4/N5/N6/N7 training or locked Haze4K command is allowed.

## Forbidden Features

- `A0_output - hazy`;
- `A0_output(aug) - A0_output(x)`;
- `teacher_output - A0_output`;
- `expert_output - anchor_output`;
- RGB output-level correction;
- locked Haze4K threshold, feature, checkpoint, or route selection.

Allowed evidence is restricted to hazy runtime scalars, ConvIR internal feature
maps, FAM response in feature space, skip-merge disagreement in feature space,
and final-feature consistency under a tiny input perturbation.

## Outputs

- `v215_s0_protocol.md`
- `v215_s0_fold_manifest.json`
- `v215_s0_forbidden_feature_contract.md`
- `v215_s0_no_training_no_locked_status.txt`
- `v215_s1_top100_overlap_report.md`
- `v215_s1_top100_hazy_vs_all_runtime.csv`
- `v215_s1_lost_severe_cases.csv`
- `v215_s1_gained_false_positive_cases.csv`
- `v215_s1_score_swap_analysis.csv`
- `v215_s1_decision.md`
- `v215_s2_spatial_feature_manifest.json`
- `v215_s2_spatial_feature_rows.csv`
- `v215_s2_spatial_feature_quality_report.md`
- `v215_s2_spatial_ablation_report.csv`
- `v215_s2_spatial_label_sensitivity.csv`
- `v215_s3_internal_sensitivity_manifest.json`
- `v215_s3_fam_response_features.csv`
- `v215_s3_skip_merge_disagreement.csv`
- `v215_s3_feature_jitter_consistency.csv`
- `v215_s3_sensitivity_ablation_report.csv`
- `v215_s4_oof_metrics.csv`
- `v215_s4_oof_predictions.csv`
- `v215_s4_pr_curves.json`
- `v215_s4_topk_enrichment.csv`
- `v215_s4_topk_overlap.csv`
- `v215_s4_bootstrap_delta.json`
- `v215_s4_fold_seed_stability.csv`
- `v215_s4_calibration_report.json`
- `v215_n1s_decision.md`

## Gates

Primary baseline: `B0_hazy_runtime_v214`.

Primary candidates: fixed B3/B5/B6/B7 groups selected from train-derived OOF
only.

Pass requires:

- severe-risk PR-AUC candidate >= hazy-runtime + `0.015`;
- paired bootstrap lower CI for PR-AUC delta is not negative;
- top-100 enrichment candidate >= hazy-runtime + `0.75`, or top100 severe
  count improves by at least `2`;
- top-50 enrichment candidate >= hazy-runtime - `0.25`;
- at least two of `-0.1/-0.2/-0.3` thresholds improve by PR-AUC or top100, and
  `-0.2` must improve;
- internal spatial or internal sensitivity improves over internal scalar;
- at least `12/15` fold-seed units do not fall below hazy-runtime top100.

Pass decision:
`N1S_SPATIAL_INTERNAL_RISK_PASS_ALLOW_V216_DESIGN_REVIEW`.

Any fail decision keeps N3/N4 blocked.

## Launch Contract

Runtime server: `convir-4090`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-15-nopost-spatial-internal-risk-audit`

Durable script:
`experience_docx/experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703/run_v215_n1s.sh`

Tmux session: `v215_n1s`

Locked Haze4K test remains untouched.

## Result

Cloud run: `convir-4090`, `2026-07-03T11:48:33+08:00` to
`2026-07-03T11:59:01+08:00`.

Source commit: `c11087f`.

Decision: `N1S_PARTIAL_INTERNAL_SIGNAL_NO_TRAINING`.

S2/S3 feature build:

- rows: `2400`;
- spatial maps: `13`;
- spatial feature columns: `1092`;
- NaN values: `0`;
- Inf values: `0`;
- locked test touched: `false`;
- training launched: `false`.

S1 top-100 decomposition:

- hazy-runtime top100 severe count: `19`;
- all-runtime top100 severe count: `15`;
- top100 overlap: `47`;
- lost severe cases: `8`;
- gained false-positive cases: `49`.

S4 primary severe-risk metrics at `WD0375_dPSNR <= -0.2`:

- baseline `B0_hazy_runtime_v214`: PR-AUC `0.149621`, ROC-AUC `0.798088`,
  top50 enrichment `10.746269`, top100 enrichment `6.805970`, top100 severe
  count `19`;
- `B3_internal_spatial`: PR-AUC `0.127275`, ROC-AUC `0.783860`,
  top100 enrichment `4.298507`, top100 severe count `12`;
- `B5_internal_sensitivity`: PR-AUC `0.132535`, ROC-AUC `0.778525`,
  top100 enrichment `4.656716`, top100 severe count `13`;
- `B7_all_runtime_spatial_sensitivity`: PR-AUC `0.093532`, ROC-AUC
  `0.728138`, top100 enrichment `4.298507`, top100 severe count `12`.

Best candidate: `B5_internal_sensitivity`.

Gate diagnostics:

- primary PR-AUC delta vs hazy-runtime: `-0.017086`;
- top100 enrichment delta vs hazy-runtime: `-2.149254`;
- top100 severe-count delta vs hazy-runtime: `-6`;
- top50 enrichment delta vs hazy-runtime: `-4.298507`;
- bootstrap PR-AUC p05: `-0.080790`;
- thresholds better: `1/3`;
- stable fold-seed units: `0/15`;
- internal spatial gain vs scalar PR-AUC: `-0.005187`;
- internal sensitivity gain vs scalar PR-AUC: `0.000073`.

Interpretation: v2.15 found a tiny sensitivity-vs-scalar signal, but not enough
to improve rare severe-risk top-tail ranking over hazy-runtime. Spatial dense
maps and feature-space sensitivity do not authorize NoPost training in this
form. N3/N4 remain blocked.

Raw S2/S3 feature tables are cloud-only because they are large intermediate
feature artifacts. Synced GitHub evidence keeps manifests, quality reports,
S1/S4 aggregates, predictions, curves, bootstrap, and closeout decisions.
