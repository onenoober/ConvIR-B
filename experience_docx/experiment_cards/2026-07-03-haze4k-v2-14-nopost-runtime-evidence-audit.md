# Haze4K v2.14 NoPost Runtime Evidence Audit

Date: 2026-07-03

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_UNTOUCHED`

Branch: `codex/haze4k-v2-14-nopost-runtime-evidence-audit`

Base route: `github/codex/haze4k-v2-13-nopost-feature-gated-adapter`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/`

## Reason

The v2.13 N1 stop was operationally correct, but its conclusion is not sealed:
the v2.13 `hazy_only` group selected every `hazy_` column, which included
`hazy_PSNR`. That value requires ground truth and is not runtime-available. The
v2.14 audit replays the N1 feature table with a strict runtime-valid feature
manifest before any N3/N4 training decision.

## Scope

This route runs only N1R, a no-training separability replay. It may reuse the
v2.13 cloud feature table because the replay filters columns and recomputes
labels from `WD0375_dPSNR`. If the table is missing on `convir-4090`, the cloud
script may rebuild it from train-core data and the WD0375 teacher cache.

Locked Haze4K test remains blocked and untouched.

## Feature Groups

- `hazy_leakcheck`: `hazy_PSNR` only; oracle leak sentinel, never a runtime gate.
- `hazy_runtime`: hazy-derived columns excluding GT/teacher metrics.
- `internal_final`: `final_*`.
- `internal_res`: `res1_*`, `res2_*`.
- `internal_scm`: `scm2_*`, `scm4_*`.
- `internal_only`: final/res/scm runtime internal features.
- `all_runtime`: `hazy_runtime + internal_only`.
- `all_with_leak`: leakcheck replay for estimating the old contamination risk.

Forbidden runtime features include `hazy_PSNR`, `A0_PSNR`, `A0_SSIM`,
`WD0375_PSNR`, `WD0375_SSIM`, `WD0375_dPSNR`, `WD0375_dSSIM`, and any GT or
teacher metric. `WD0375_dPSNR` may only define offline labels.

## Outputs

- `v214_n1r_runtime_feature_manifest.json`
- `v214_n1r_leakage_report.md`
- `v214_n1r_oof_metrics.csv`
- `v214_n1r_oof_predictions.csv`
- `v214_n1r_delta_auc_bootstrap.json`
- `v214_n1r_topk_risk_enrichment.csv`
- `v214_n1r_internal_block_ablation.csv`
- `v214_n1r_label_sensitivity.csv`
- `v214_n1r_decision.md`

## Gates

N1R passes only if:

- benefit `all_runtime` ROC-AUC >= `0.70`;
- severe-risk `all_runtime` ROC-AUC >= `0.70`;
- severe-risk `internal_only` ROC-AUC >= `hazy_runtime - 0.01`;
- severe-risk `all_runtime` improves PR-AUC or top-k risk enrichment over
  `hazy_runtime`;
- paired bootstrap does not show `internal_only` significantly worse than
  `hazy_runtime`.

Pass decision: `N1R_RUNTIME_EVIDENCE_PASS_ALLOW_N3_DESIGN_REVIEW`.

Failure with strong leakcheck evidence: recommend N1S spatial/internal expansion
and do not train.

Failure without leak dominance: current evidence is insufficient and do not
train.

## Launch Contract

Runtime server: `convir-4090`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-14-nopost-runtime-evidence-audit`

Durable script:
`experience_docx/experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/run_v214_n1r.sh`

Tmux session: `v214_n1r`

No training, evaluation on locked test, inference demo, checkpoint write, or
model selection is allowed in this step.

## Result

Cloud run: `convir-4090`, `2026-07-03T10:43:16+08:00` to
`2026-07-03T10:44:02+08:00`.

Source commit: `0704c89`.

Decision: `N1R_RUNTIME_EVIDENCE_FAIL_INSUFFICIENT_NO_TRAINING`.

The replay reused the v2.13 cloud-only feature table:

```text
/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv
```

Locked Haze4K test remained untouched.

Primary runtime-valid results:

- rows: `2400`;
- benefit positives: `2266`;
- severe-risk positives: `67`;
- benefit all-runtime ROC-AUC: `0.811898`;
- severe-risk all-runtime ROC-AUC: `0.826237`;
- benefit internal/runtime-hazy ROC-AUC: `0.802239` / `0.770909`;
- severe-risk internal/runtime-hazy ROC-AUC: `0.819616` / `0.798088`;
- severe-risk all-runtime PR-AUC: `0.135348`;
- severe-risk runtime-hazy PR-AUC: `0.149621`;
- severe-risk all-runtime minus runtime-hazy PR-AUC: `-0.014273`;
- severe-risk all-runtime top-100 enrichment: `5.373134`;
- severe-risk runtime-hazy top-100 enrichment: `6.805970`;
- severe-risk all-runtime minus runtime-hazy top-100 enrichment: `-1.432836`;
- bootstrap worse-than-margin findings: `0`.

Leakage check:

- `hazy_PSNR` was found and excluded from runtime groups;
- severe-risk leakcheck `hazy_PSNR` ROC-AUC: `0.698620`;
- severe-risk all-with-leak ROC-AUC: `0.824894`;
- severe-risk all-runtime ROC-AUC: `0.826237`.

Interpretation: removing `hazy_PSNR` changes the v2.13 conclusion, because
runtime-valid internal features are no longer worse than runtime-hazy features
by ROC-AUC. However, the severe-risk prioritization gate still fails: the
combined runtime feature set does not improve PR-AUC or top-k risk enrichment
over hazy-runtime features. This is insufficient evidence for N3/N4 training.

No N3, N4, N5, N6, or N7 command was launched.
