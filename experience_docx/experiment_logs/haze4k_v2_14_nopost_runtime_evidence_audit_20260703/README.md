# Haze4K v2.14 NoPost Runtime Evidence Audit

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_UNTOUCHED`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-14-nopost-runtime-evidence-audit`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Previous feature table source:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-13-nopost-feature-gated-adapter/experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/v213_n1_feature_rows_cloud_only.csv`

Locked Haze4K test: untouched and blocked.

## Plan

Run N1R only. The replay filters out GT/teacher-derived columns, writes a
runtime feature manifest, reports `hazy_PSNR` as a leakcheck sentinel, and emits
OOF ROC-AUC, PR-AUC, ECE, bootstrap delta AUC, top-k severe-risk enrichment,
internal block ablations, label sensitivity, and a decision note.

No N3/N4 training is launched by this route.

## Expected Outputs

- `v214_n1r_runtime_feature_manifest.json`
- `v214_n1r_leakage_report.md`
- `v214_n1r_oof_metrics.csv`
- `v214_n1r_oof_predictions.csv`
- `v214_n1r_delta_auc_bootstrap.json`
- `v214_n1r_topk_risk_enrichment.csv`
- `v214_n1r_internal_block_ablation.csv`
- `v214_n1r_label_sensitivity.csv`
- `v214_n1r_decision.md`

## Closeout

Cloud run: `convir-4090`, `2026-07-03T10:43:16+08:00` to
`2026-07-03T10:44:02+08:00`.

Source commit: `0704c89`.

Decision: `N1R_RUNTIME_EVIDENCE_FAIL_INSUFFICIENT_NO_TRAINING`.

The run reused the v2.13 cloud-only feature table and did not rebuild raw
features. Locked Haze4K test remained untouched.

Key results:

- rows: `2400`;
- benefit all-runtime ROC-AUC: `0.811898`;
- severe-risk all-runtime ROC-AUC: `0.826237`;
- benefit internal/runtime-hazy ROC-AUC: `0.802239` / `0.770909`;
- severe-risk internal/runtime-hazy ROC-AUC: `0.819616` / `0.798088`;
- severe-risk all-runtime/runtime-hazy PR-AUC: `0.135348` / `0.149621`;
- severe-risk all-runtime/runtime-hazy top-100 enrichment: `5.373134` /
  `6.805970`;
- severe-risk all-runtime minus runtime-hazy PR-AUC: `-0.014273`;
- severe-risk all-runtime minus runtime-hazy top-100 enrichment: `-1.432836`.

The `hazy_PSNR` leakcheck column was present but excluded from runtime groups.
It did not rescue the gate: all-with-leak severe-risk ROC-AUC was `0.824894`,
while all-runtime severe-risk ROC-AUC was `0.826237`.

Conclusion: v2.13's old hazy-only comparison was contaminated by a GT-derived
feature, but v2.14 still does not authorize training. The runtime-valid evidence
is strong enough for AUC separability but not for severe-risk prioritization.
No N3/N4 training was launched.
