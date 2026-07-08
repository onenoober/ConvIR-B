# CHD-RM Haze4K Experiment Index

Date: 2026-07-08

Status: v2 density-need calibration prepared; cloud execution pending.

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
| v2 density-need calibration | `codex/haze4k-v5-v2-chd-rm-density-need-calibration` | prepared | D0/D1/D3/D4/D5 authorized on train_inner/val_inner only | cloud run pending | `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/` |
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
