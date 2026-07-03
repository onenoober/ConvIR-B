# Haze4K v2.14 NoPost Runtime Evidence Audit

Status: `PLANNED_N1R_ONLY_LOCKED_TEST_UNTOUCHED`

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
