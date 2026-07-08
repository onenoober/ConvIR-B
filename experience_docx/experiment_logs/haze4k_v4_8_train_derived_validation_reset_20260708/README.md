# Haze4K v4.8 Train-Derived Validation Reset Evidence

Route card: `experience_docx/experiment_cards/2026-07-08-haze4k-v4-8-train-derived-validation-reset.md`

Central index: `experience_docx/EXPERIMENT_INDEX.md`

Route id: `haze4k_v4_8_train_derived_validation_reset_20260708`

Status: `COMPLETED_GATE_FAIL`.

Decision: do not promote the DCFSB-bottleneck adapter recipe; do not run locked test; do not expand v4.5/SDC-Lite or connect this R field to skip/FAM/restoration outputs.

## Scope

Phase 1 trained five train-derived folds from official A0 using the DCFSB-bottleneck adapter-only recipe. An initial `batch_size=8` launch exposed an engineering-only failure in fold2/fold4 because their final training batch had size 1; all five folds were rerun with the predeclared `batch_size=7` repair for consistent OOF comparison.

Phase 2 ran an R-only calibration audit using the existing v4.5 SDC-Lite checkpoint on the same 3000 train-derived OOF union. Restoration outputs were ignored. No training, prediction images, locked-test access, or test split enumeration occurred.

## Phase 1 OOF Summary

- Count: `3000` train-derived held-out images.
- A0 mean PSNR: `39.391236`.
- Candidate mean PSNR: `39.442427`.
- Mean dPSNR: `0.051192`.
- Positive ratio: `0.624667`.
- Median dPSNR: `0.050976`.
- p5 / p1 dPSNR: `-0.266470` / `-0.469765`.
- Bootstrap 95% CI: `[0.041783, 0.059540]`.
- Sign-test one-sided p: `3.3161829e-43`.
- Mean dHighL1: `0.0000017954`.
- Failed OOF gates: `oof_p5_delta_psnr, every_proxy_bin_positive_ratio, low_saturation_subgroup`.
- Low-saturation q1 mean dPSNR / positive ratio: `0.010802` / `0.488000`.

Interpretation: the recipe has a real positive train-derived OOF signal, but it does not pass the full tail-safe gate because p5 is too negative and low-saturation coverage remains below the required win rate.

## Phase 2 R-only Summary

- R mean: `0.491335`.
- R std mean: `0.081058`.
- corr(R, input-GT L1): `-0.438983`.
- corr(R, A0 low+high error proxy): `-0.421714`.
- heavy-haze q4 vs q1 relative R response: `-0.059567`.
- A0-error q4 vs q1 relative R response: `-0.066312`.
- low-saturation q1 minus high-saturation q4 R mean: `-0.022944`.
- Failed R-only gates: `R_1_2_std_mean, corr_R_input_gt_l1, corr_R_a0_error_proxy, corr_R_dark_channel_direction, heavy_haze_q4_gt_q1_by_10pct, a0_error_q4_gt_q1_by_10pct, low_saturation_no_reverse`.

Interpretation: the existing SDC-Lite R field is not a calibrated haze/error controller. It is low-variance and directionally reversed against the key haze/error proxies, so SDC-Lite v2 and skip/FAM/restoration modulation using this R are blocked.

## Primary Evidence Files

- `v48_closeout.json`: final route closeout covering Phase 1 and Phase 2.
- `decision_after_v48.md`: Phase 1 OOF decision.
- `v48_oof_summary.json`: OOF metrics, gates, failed gates, fold summaries, proxy failures.
- `v48_oof_bootstrap.json` and `v48_oof_sign_test.json`: aggregate statistical checks.
- `v48_oof_proxy_bins.csv`: aggregate proxy-bin audit.
- `v48_fold_summaries.csv`: fold-level metrics.
- `v48_oof_worst64_compact.csv`: compact worst-tail sample table.
- `r_only_calibration_sdc_lite_oof3000/r_only_summary.json`: R-only closeout.
- `r_only_calibration_sdc_lite_oof3000/decision_after_v48_r_only.md`: R-only decision.
- `r_only_calibration_sdc_lite_oof3000/r_only_proxy_bins.csv`: R-only proxy-bin response audit.
- `v48_bs7_engineering_repair_note.md`: engineering repair note for the batch-size-1 BatchNorm issue.

Cloud-only broad tables not synced to GitHub main by default: `v48_oof_per_image_compact.csv`, fold-level `val_per_image_compact.csv`, and `r_only_calibration_sdc_lite_oof3000/r_only_stats_compact.csv`.
