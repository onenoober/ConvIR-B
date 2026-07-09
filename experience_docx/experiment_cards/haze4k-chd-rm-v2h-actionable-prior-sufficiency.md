# Haze4K CHD-RM v2h Actionable Prior Sufficiency

Status: `COMPLETED_WITH_D_PREFLIGHT_BLOCKED`

Decision label: `V2H_ABC_PASS_D_BLOCKED_CREATE_SEPARATE_NOOP_ARCH_BRANCH`

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

## A/B Closeout

v2h-A and v2h-B both passed under the internal train/val contract. No training,
locked Haze4K test, D2, F5, v3, RARM connection, RARM training, adapter training,
or canary expansion was run.

- v2h-A D7c fixed operating point: coverage `0.302695`, action recall
  `0.548312`, low-adjacent recall `0.155904`, negative false rate `0.002974`,
  isolated-LDHN hit rate `0.022366`, and per-image negative false p95
  `0.047619`.
- Density-matched control at comparable coverage: action recall `0.448391` and
  negative false rate `0.047786`.
- v2h-B alpha `0.3` D7c shadow-modulation upper bound: global PSNR gain
  `1.374164`, action-region PSNR gain `1.695614`, negative touch rate
  `0.002698`, isolated touch rate `0.023606`.
- Density-matched alpha `0.3` global PSNR gain: `0.977430`.
- Action-oracle alpha `0.3` global PSNR gain: `2.220821`.

Conclusion: the immediate bottleneck is no longer whether a deployable
actionable prior exists. D7c is sufficient to justify only v2h-C OOF stability
and v2h-D FAM2 no-op equivalence review. RARM/training/locked-test access remain
blocked.

## C/D Closeout

v2h-C passed fold calibration stability with no training and no locked test:

- D7c calibrated action recall mean/min `0.576335` / `0.556955`;
- low-adjacent recall mean `0.170063`;
- negative false mean/max `0.003403` / `0.003996`;
- selected coverage std `0.010785`;
- density-matched negative false mean/max `0.049636` / `0.063885`.

v2h-D was correctly blocked before numerical no-op equivalence because the v2h
branch preserves the official architecture anchor:

```text
Official ConvIR-B anchor only supports fam_mode='original'. Create a route branch for architecture variants.
```

Conclusion: D7c prior sufficiency is supported by A/B/C. FAM2/no-op insertion
must move to a separate model-structure no-op branch from
`github/codex/haze4k-official-arch-anchor`; do not mutate v2h into an
architecture branch. Locked test, D2/F5/v3, RARM connection/training, adapter
training, canary expansion, and architecture mutation inside v2h remain blocked.
