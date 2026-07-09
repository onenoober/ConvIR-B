# CHD-RM Haze4K Experiment Index

Date: 2026-07-09

Status: v2d D7c top-k is promising, but controls are weak; pause before v3/RARM.

## Research Direction

```text
连续雾浓度感知的区域自适应残差调制与低雾区域保护去雾方法研究
```

## Current Scope

- Dataset: Haze4K only.
- Backbone: ConvIR-B.
- Task: single-image dehazing.
- Route family: CHD-RM v5.
- Test policy: Haze4K locked test is final-confirmation only.

## Invariants

1. Start from `github/codex/haze4k-official-arch-anchor`.
2. Keep the route inside research content one: continuous haze-density-aware
   region-adaptive residual modulation with low-haze protection.
3. Do not turn the route into independent color, luminance, texture, or
   structure modeling.
4. Do not use Lab, luminance, gradient, or texture as core training targets.
5. Do not replace the ConvIR-B backbone in this route.
6. Do not connect or train RARM before density/need calibration and no-op
   equivalence gates pass.
7. Do not use Haze4K locked test for checkpoint, threshold, route, scale,
   gamma, mask, loss, or hyperparameter selection.
8. Any candidate claim must beat matched-budget controls, not only A0.
9. Any final candidate must preserve low-haze regions and tail safety.

## Route Table

| Stage | Branch | Status | Main Result | Decision | Evidence Root |
| --- | --- | --- | --- | --- | --- |
| v0 route lock | `codex/haze4k-v5-v0-chd-rm-route-lock` | completed | route scope, locked-test policy, stages, and archive paths fixed | proceed to v1 preflight only | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v0_route_lock_20260708/` |
| v1 data baseline lock | `codex/haze4k-v5-v1-chd-rm-data-baseline-lock` | completed | train/test manifest, 2400/600 split, OOF folds, A0 val600, metric reproducibility, and efficiency locked | `COMPLETED_V1_GATE_PASS` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/` |
| v2 density-need calibration | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` | paused | density passes strongly; need remains below gate; shuffled control fails | `PAUSE_V2_DUAL_HEAD_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/` |
| v2b need calibration repair | `codex/haze4k-v5-v2b-chd-rm-need-calibration-repair` | paused | D6c improves need ranking (Pearson 0.3632, Spearman 0.3298, AUROC 0.7133) and shuffled control fails, but strong-response coverage remains 0 | `PAUSE_V2B_NEED_REPAIR_NOT_PASSED` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2b_need_calibration_repair_20260708/` |
| v2c need coverage calibration | `codex/haze4k-v5-v2c-chd-rm-need-coverage-calibration` | paused | Train-inner monotone/affine calibration restores coverage but creates unsafe false-strong responses; best d6c coverage-safe candidate still exceeds false-strong gate (0.1153 > 0.10) | `PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2c_need_coverage_calibration_20260709/` |
| v2d need spatial hard-negative | `codex/haze4k-v5-v2d-chd-rm-need-spatial-hard-negative` | paused | D7c frozen multi-context top-k HN passes candidate safety (Spearman 0.5175, AUROC 0.8456, coverage 0.3027, false p90 0.0246), but shuffled/random controls retain weak proxy signal | `PAUSE_V2D_D7C_TOPK_PROMISING_BUT_CONTROLS_WEAK_NO_V3` | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/` |
| v3 no-op RARM audit | `codex/haze4k-v5-v3-chd-rm-noop-rarm-audit` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3_noop_rarm_audit_20260708/` |
| v4 single-scale RARM | `codex/haze4k-v5-v4-chd-rm-single-scale-rarm` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v4_single_scale_rarm_20260708/` |
| v5 low-haze protection | `codex/haze4k-v5-v5-chd-rm-low-haze-protection` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v5_low_haze_protection_20260708/` |
| v6 multiscale haze modulation | `codex/haze4k-v5-v6-chd-rm-multiscale-haze-modulation` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v6_multiscale_haze_modulation_20260708/` |
| v7 OOF candidate lock | `codex/haze4k-v5-v7-chd-rm-oof-candidate-lock` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v7_oof_candidate_lock_20260708/` |
| v8 final Haze4K confirmation | `codex/haze4k-v5-v8-chd-rm-final-haze4k-confirmation` | planned | TBD | TBD | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v8_final_haze4k_confirmation_20260708/` |

## Gate Summary

| Stage | Must Pass Before |
| --- | --- |
| v1 data/baseline lock | any density/need training |
| v2 density/need calibration | RARM connection |
| v3 no-op RARM audit | RARM training |
| v4 single-scale matched controls | final candidate consideration |
| v5 low-haze protection | final candidate consideration |
| v6 multiscale check | selecting CHD-RM-MS over CHD-RM-LP |
| v7 OOF candidate lock | any Haze4K locked-test command |
| v8 final confirmation | promotion wording |

## Metric Families

- restoration quality: PSNR, SSIM, LPIPS, dPSNR, dSSIM, dLPIPS;
- region quality: low/medium/heavy/very-heavy haze PSNR and dPSNR;
- statistics: mean, median, positive ratio, p5, p10, CVaR5, worst32,
  bootstrap CI, sign-test p;
- calibration: density/need Pearson, Spearman, AUROC, AUPRC, monotonicity,
  mask coverage, false strong-recovery rate;
- modulation: gamma means by bucket, gamma correlations, residual norms;
- efficiency: params, FLOPs, FPS, latency, peak GPU memory, training time.

## Pause Rules

- Pause immediately if a required asset, split, checkpoint, or Python path is
  missing.
- Pause if a command tries to touch locked test before v7 candidate lock.
- Pause if local-only execution would be needed for runtime validation.
- Pause if a stage gate fails and the next step would expand scope instead of
  diagnosing the failed gate.
