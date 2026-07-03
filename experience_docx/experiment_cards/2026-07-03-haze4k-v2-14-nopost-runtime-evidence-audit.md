# Haze4K v2.14 NoPost Runtime Evidence Audit

Date: 2026-07-03

Status: `PLANNED_N1R_ONLY_LOCKED_TEST_UNTOUCHED`

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
