# CHD-RM Haze4K Experiment Index

Date: 2026-07-09

Status: v2e completed; v2f first-stage supports a frozen-side density-stratified head canary. v3/RARM remains blocked.

## Research Direction

```text
Continuous haze-density-aware region-adaptive residual modulation with low-haze protection for ConvIR-B Haze4K dehazing.
```

## Current Scope

- Dataset: Haze4K only.
- Backbone: ConvIR-B.
- Task: single-image dehazing.
- Route family: CHD-RM v5.
- Test policy: Haze4K locked test is final-confirmation only.

## Invariants

1. Start from `github/codex/haze4k-official-arch-anchor`.
2. Keep the route inside research content one: continuous haze-density-aware region-adaptive residual modulation with low-haze protection.
3. Do not turn the route into independent color, luminance, texture, or structure modeling.
4. Do not use Lab, luminance, gradient, or texture as core training targets.
5. Do not replace the ConvIR-B backbone in this route.
6. Do not connect or train RARM before density/need calibration, control, recall-protection, and no-op equivalence gates pass.
7. Do not use Haze4K locked test for checkpoint, threshold, route, scale, gamma, mask, loss, or hyperparameter selection.
8. Any candidate claim must beat matched-budget controls, not only A0.
9. Any final candidate must preserve low-haze regions and tail safety.

## Route Table

| Stage | Branch | Status | Main Result | Decision | Evidence Root |
| --- | --- | --- | --- | --- | --- |
| v0 route lock | `codex/haze4k-v5-v0-chd-rm-route-lock` | completed | route scope, locked-test policy, stages, and archive paths fixed | `COMPLETED_V0_ROUTE_LOCK` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v0_route_lock_20260708/` |
| v1 data baseline lock | `codex/haze4k-v5-v1-chd-rm-data-baseline-lock` | completed | train/test manifest, 2400/600 split, OOF folds, A0 val600, metric reproducibility, and efficiency locked | `COMPLETED_V1_GATE_PASS` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/` |
| v2 density-need calibration | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` | paused | density passes strongly; need remains below gate; shuffled control fails | `PAUSE_V2_DUAL_HEAD_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/` |
| v2b need calibration repair | `codex/haze4k-v5-v2b-chd-rm-need-calibration-repair` | paused | D6c improves need ranking but strong-response coverage remains 0 | `PAUSE_V2B_NEED_REPAIR_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/` |
| v2c need coverage calibration | `codex/haze4k-v5-v2c-chd-rm-need-coverage-calibration` | paused | Train-inner calibration restores coverage but creates unsafe false-strong responses | `PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2c_need_coverage_calibration_20260709/` |
| v2d need spatial hard-negative | `codex/haze4k-v5-v2d-chd-rm-need-spatial-hard-negative` | paused | D7c frozen multi-context top-k HN is promising, but controls remained weak | `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/` |
| v2e D7c control recall audit | `codex/haze4k-v5-v2e-chd-rm-d7c-control-recall-audit` | paused | Fixed permutation and density matched controls are clean, but D7c top-k LDHN recall is low and D7c-RP has no safe recall-protected point | `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2e_d7c_control_recall_audit_20260709/` |
| v2f need target/head redesign | `codex/haze4k-v5-v2f-chd-rm-need-target-head-redesign` | F4 authorized | F0-F3/F2 first-stage shows LDHN core support, frozen feature separability, and density-conditioned target de-proxying; run frozen-side density-stratified head canary only | `F4_AUTHORIZED_PENDING_CLOUD_NO_V3_RARM` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2f_need_target_head_redesign_20260709/` |
| v2h actionable prior sufficiency | `codex/haze4k-v5-v2h-actionable-prior-sufficiency` | planned | Test whether D7c has a stable conservative operating band before shadow-modulation/no-op work | `PLANNED_V2H_ACTIONABLE_PRIOR_SUFFICIENCY_AUDIT` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709/` |
| v3 no-op RARM audit | `codex/haze4k-v5-v3-chd-rm-noop-rarm-audit` | blocked | blocked by v2e RP failure | `BLOCKED_BY_V2E_D7C_RP_NO_SAFE_POINT` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3_noop_rarm_audit_20260708/` |
| v4 single-scale RARM | `codex/haze4k-v5-v4-chd-rm-single-scale-rarm` | blocked | blocked until v3 no-op gate is authorized and passed | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4_single_scale_rarm_20260708/` |
| v5 low-haze protection | `codex/haze4k-v5-v5-chd-rm-low-haze-protection` | blocked | blocked until a safe R_need/RARM gate exists | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v5_low_haze_protection_20260708/` |
| v6 multiscale haze modulation | `codex/haze4k-v5-v6-chd-rm-multiscale-haze-modulation` | blocked | blocked until earlier gates pass | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v6_multiscale_haze_modulation_20260708/` |
| v7 OOF candidate lock | `codex/haze4k-v5-v7-chd-rm-oof-candidate-lock` | blocked | blocked until candidate exists | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v7_oof_candidate_lock_20260708/` |
| v8 final Haze4K confirmation | `codex/haze4k-v5-v8-chd-rm-final-haze4k-confirmation` | blocked | blocked until v7 candidate lock | `BLOCKED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v8_final_haze4k_confirmation_20260708/` |

## v2e Closeout

D7c top-k should be retained as evidence of real `R_need` ranking signal, not promoted to RARM. v2e fixed permutation and density-only matched controls are clean. The remaining blocker is safety/recall incompatibility:

- frozen D7c top-k LDHN recall `0.0370` with safe false-tail;
- first LDHN-passing RP point LDHN recall `0.1096` but false-p90 `0.0599` and false-p95 `0.2069`;
- strongest RP point LDHN recall `0.1822` but false-p95 `0.5348`.

Current next action is not v3/RARM. Any continuation must diagnose or redesign the frozen-side-head `R_need` target/head so LDHN recall and false-tail safety pass together.

## v2f First-Stage Decision

v2f F0-F3/F2 completed on `convir-4090` without D2, v3, RARM, ConvIR-B
unfreeze, or locked Haze4K test. First-stage evidence supports a bounded F4
canary:

- LDHN pixel coverage `0.08988972981770833`.
- LDHN core fraction of LDHN `0.569798970635499`.
- LDHN unstable fraction of LDHN `0.04701398288013833`.
- Best frozen feature probe `feature_set_2` + `mlp`: AUROC
  `0.8107264347671554`, AUPRC `0.807792756659645`.
- Density-conditioned target density Spearman `0.007215705298292346`, compared
  with global target density Spearman `0.31464418569286756`.

Authorized next action: F4 density-stratified frozen-side `R_need` head canary
on `train_inner`/`val_inner`. F4 does not authorize v3/RARM. If F4 passes, the
next required phase is F5 stricter controls before any v3 no-op audit.

## v2h A/B Closeout

v2h tested whether D7c is sufficient as a deployable actionable prior before any
future no-op/RARM/adapter work. It ran only risk-coverage and diagnostic
shadow-modulation audits on the internal Haze4K split. No locked test, D2, F5,
v3, RARM connection, RARM training, adapter training, new head family, or canary
expansion was run.

v2h-A fixed D7c operating point:

- coverage `0.302695`;
- action recall `0.548312`;
- low-adjacent recall `0.155904`;
- negative false rate `0.002974`;
- isolated-LDHN hit rate `0.022366`;
- per-image negative false p95 `0.047619`.

The density-matched control at comparable coverage was worse: action recall
`0.448391` and negative false rate `0.047786`.

v2h-B alpha `0.3` shadow-modulation diagnostic:

- D7c global PSNR gain `1.374164`;
- density-matched global PSNR gain `0.977430`;
- action-oracle global PSNR gain `2.220821`;
- D7c action-region gain `1.695614`;
- D7c negative touch `0.002698`;
- D7c isolated touch `0.023606`.

Decision: `V2H_AB_PASS_PRIOR_SUFFICIENT_AUTHORIZE_OOF_AND_NOOP_ONLY`. D7c is sufficient to justify v2h-C OOF stability and
v2h-D FAM2 no-op equivalence review only. The remaining bottleneck is connection
risk, not deployable-prior existence. RARM/training/locked-test access remain
blocked.

## v2h C/D Closeout

v2h-C ran the authorized no-training fold calibration stability audit over the
v1 fixed five-fold train OOF table. D7c stayed stable and safer than density
matching:

- D7c action recall mean/min `0.576335` / `0.556955`;
- D7c low-adjacent recall mean `0.170063`;
- D7c negative false mean/max `0.003403` / `0.003996`;
- D7c selected coverage std `0.010785`;
- density-matched negative false mean/max `0.049636` / `0.063885`.

v2h-D attempted the authorized FAM2/no-op equivalence review but stopped at the
correct preflight boundary: the v2h branch preserves the official architecture
anchor and rejects `fam2_modres`. This means no-op insertion must be designed on
a separate model-structure branch from `github/codex/haze4k-official-arch-anchor`.
It does not authorize RARM/training/locked-test access.

Decision: `V2H_ABC_PASS_D_BLOCKED_CREATE_SEPARATE_NOOP_ARCH_BRANCH`.

## Gate Summary

| Stage | Must Pass Before |
| --- | --- |
| v1 data/baseline lock | any density/need training |
| v2 density/need calibration | RARM connection |
| v2e control and recall audit | v3 no-op RARM audit |
| v3 no-op RARM audit | RARM training |
| v4 single-scale matched controls | final candidate consideration |
| v5 low-haze protection | final candidate consideration |
| v6 multiscale check | selecting CHD-RM-MS over CHD-RM-LP |
| v7 OOF candidate lock | any Haze4K locked-test command |
| v8 final confirmation | promotion wording |

## Metric Families

- restoration quality: PSNR, SSIM, LPIPS, dPSNR, dSSIM, dLPIPS;
- region quality: low/medium/heavy/very-heavy haze PSNR and dPSNR;
- statistics: mean, median, positive ratio, p5, p10, CVaR5, worst32, bootstrap CI, sign-test p;
- calibration: density/need Pearson, Spearman, AUROC, AUPRC, monotonicity, mask coverage, false strong-recovery rate;
- modulation: gamma means by bucket, gamma correlations, residual norms;
- efficiency: params, FLOPs, FPS, latency, peak GPU memory, training time.

## Pause Rules

- Pause immediately if a required asset, split, checkpoint, or Python path is missing.
- Pause if a command tries to touch locked test before v7 candidate lock.
- Pause if local-only execution would be needed for runtime validation.
- Pause if a stage gate fails and the next step would expand scope instead of diagnosing the failed gate.
