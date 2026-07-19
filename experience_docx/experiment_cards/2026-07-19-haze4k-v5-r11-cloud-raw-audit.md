# Haze4K v5 R11 Cloud Raw Evidence Audit

Date: 2026-07-19

Status: COMPLETED POST-HOC READ-ONLY AUDIT

## Identity

- Audit id: `haze4k_v5_r11_cloud_raw_audit_20260719`.
- Audited experiment: `haze4k_v5_r11_regional_action_observability_20260719`.
- Evidence cutoff: `github/main@893ba97790ad19d745ff676f5bbf28bd37395d50`.
- Source route commit: `c183817e2b3befdeeb12278aa6e6a0574883b6d5`.
- Source run: `r11-a0-regional-observability-r1`.
- Audit run: `r11-cloud-raw-audit-r2`; r1 was superseded only to remove a
  NumPy empty-slice warning from an optional post-hoc field.
- Protected-data policy: confirmation identities, targets and outcomes, canary
  and locked test were prohibited and untouched.

## Scope And Decision Boundary

The audit used only the R11 route checkout, its eight whitelisted development
assets, the formal 384-row per-image replay table and the 49,152-row tile/action
prediction table. It verified source/config/input identities, re-ran the exact
fold-stratified 4,000-draw bootstrap, reconstructed the frozen exact R10 action
budget, and produced descriptive action, margin, calibration, risk-coverage,
label-threshold sensitivity and 8x8 position summaries. It did not train or
infer a restoration model,
change a threshold, choose a seed/checkpoint/sample, search a neighboring
feature or access protected roles.

This is post-hoc exploratory development evidence. It may explain but cannot
rewrite `COMPLETED_GATE_FAIL / R11_A0_REGIONAL_OBSERVABILITY_FAIL_STOP / NONE`.
The audit decision is
`R11_ORIGINAL_BOTTLENECK_SUPPORTED_BUT_NARROWED / NONE`.

## Evidence Contract

- Identity: all eight whitelisted assets must exist; every file hash must match;
  all 768 R11 cache units must match their manifest hashes; R11 population must
  be 384 unique names, 192 per fold, 128 tile/action rows per name.
- Reproduction: every official bootstrap point/LCB95/UCB95 must reproduce to
  absolute tolerance `1e-12` from the cloud-only per-image table.
- Descriptive analyses: image-gain concentration, reconstructed action accuracy,
  action-specific harm, score calibration, risk-coverage, top-score harm,
  top-1/top-2 margins and fixed-grid position counts.
- Allowed outcome: preserve or narrow the existing bottleneck and mark preliminary
  questions answered/modified/deferred.
- Prohibited outcome: change the R11 terminal, authorize continuation, claim
  deployment/external validity, or infer fog severity/semantics without labels.

## Artifact Boundary

GitHub retains only compact JSON/CSV audit summaries, this record, one README,
one typed audit closeout and one conclusion. The original per-image table,
tile/action table, cache tensors, dataset, logs and audit scripts remain
cloud-only.
