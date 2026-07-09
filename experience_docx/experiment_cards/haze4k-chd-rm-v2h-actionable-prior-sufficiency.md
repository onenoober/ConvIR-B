# Haze4K CHD-RM v2h Actionable Prior Sufficiency

Status: `PLANNED`

Decision label: `PLANNED_V2H_ACTIONABLE_PRIOR_SUFFICIENCY_AUDIT`

Evidence root: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709/`.

## Route Identity

v2h is a diagnostic continuation after v2g. It does not train a new head, does not connect or train RARM, and does not use Haze4K locked test. Its purpose is to test whether D7c, the current best deployable actionable prior under the v2g three-state target, is sufficient as a conservative selector before any future no-op RARM or bounded adapter work is authorized.

## Fact Sources

- GitHub `main` compact evidence: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.
- Parent route evidence: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2g_need_actionability_audit_20260709/`.
- Cloud runtime/raw source: `convir-4090` under `/sda/home/wangyuxin/ConvIR-B/repos/`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Source And Assets

- Starting commit: parent v2g route commit `044b779`.
- Branch: `codex/haze4k-v5-v2h-actionable-prior-sufficiency`.
- Dataset: Haze4K train split internal `2400/600`; val-inner is report-only.
- Baseline checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- D3 density artifact: previous v2 density-only head.
- D7c artifact: previous v2d full D7c top-k hard-negative ordinal head.

## Forbidden Flows

- No Haze4K locked test.
- No D2, F5, v3, RARM connection, RARM training, or adapter training.
- No new selective probe/head family.
- No threshold, alpha, checkpoint, or route selection from locked test.
- No broad queue or canary expansion before the written v2h A/B gates close.

## Metric Contract

v2h-A evaluates D7c risk-coverage under the v2g three-state target. Thresholds are selected on a fixed train-calib subset; val-inner is report-only.

v2h-B, if v2h-A passes, evaluates shadow-modulation upper bound:

```text
A0 = ConvIR-B baseline output
R_oracle = GT - A0
M = calibrated selector mask
Y_shadow(alpha) = A0 + alpha * M * R_oracle
alpha in {0.1, 0.2, 0.3, 0.5, 1.0}
```

Shadow results are diagnostic upper bounds, not deployable model metrics.

## Gate

Continue to v2h-B only if v2h-A finds a stable D7c operating point and density-only controls remain weaker. Continue to v2h-C/D only if A/B jointly justify it.
