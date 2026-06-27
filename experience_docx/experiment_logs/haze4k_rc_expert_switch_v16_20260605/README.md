# Haze4K v1.6 Risk-Calibrated Expert Switch Evidence

Status: `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`; cloud intermediate analysis
and one-shot locked-test confirmation completed on `dehaze1`.

Route card:

- `experience_docx/experiment_cards/2026-06-05-haze4k-convir-v1-6-rc-expert-switch.md`

Primary files:

- `run_v16_rcswitch_intermediate_analysis.sh`: cloud launcher.
- `status.txt`: cloud status markers.
- `offline_intermediate_analysis/route_utility_leaderboard.csv`: unified
  retrospective Mechanism / Utility / Promotion gate table.
- `offline_intermediate_analysis/hard_expert_leaderboard.csv`: hard expert
  candidate ranking.
- `offline_intermediate_analysis/global_model_leaderboard.csv`: strict
  promotion-style ranking.
- `offline_intermediate_analysis/utility_tradeoff_leaderboard.csv`: risk-adjusted
  utility ranking.
- `offline_intermediate_analysis/expert_bank_oracle_switch_a0_udp.json`: A0+UDP
  oracle-switch upper bound.
- `offline_intermediate_analysis/expert_bank_oracle_switch_a0_udp_fam2.csv`:
  optional A0+UDP+FAM2 overlap oracle diagnostic.
- `offline_intermediate_analysis/udp_accept_label_predictability_oof.csv`:
  first accept-label predictability diagnostic.
- `offline_intermediate_analysis/udp_bad_risk_predictability_oof.csv`: first
  bad-risk predictability diagnostic.
- `offline_intermediate_analysis/rc_expert_switch_oof_summary.json`: true
  5-fold threshold-switch OOF diagnostic.
- `offline_intermediate_analysis/rc_expert_switch_fixed_policy_candidate.json`:
  fixed internal candidate derived from fold-selected thresholds.
- `udp_switch_features/udp_switch_feature_table.csv`: reusable per-image feature
  table for later router calibration.
- `locked_test_fixed_policy/rcswitch_locked_test_summary.json`: one-shot locked
  Haze4K test result for the fixed policy.
- `locked_test_fixed_policy/rcswitch_locked_test_per_image.csv`: locked-test
  per-image table.
- `locked_test_fixed_policy/rcswitch_locked_test_failure_audit.csv`:
  locked-test regression audit.

## Internal Result

- Retrospective leaderboard wrote 17 route summaries with 0 missing sources.
- A0+UDP GT oracle upper bound passed strongly: mean `+0.741695 dB`, hard
  bottom-25 `+1.003794 dB`, easy top-25 `+0.595787 dB`, SSIM delta
  `+0.000230`, strong regression ratio `0`, worst regression ratio `0`.
- True 5-fold OOF threshold switch over `udp_switch_feature_table` passed the
  internal Utility and Promotion-style gates: mean `+0.235332 dB`, hard
  bottom-25 `+0.512663 dB`, easy top-25 `+0.055742 dB`, SSIM delta
  `+0.000095`, coverage `0.195`, strong regression ratio `0.066667`, worst
  regression ratio `0.046667`.
- Fixed median-threshold candidate:
  `udp_a0_luma_shift_mean <= -0.003969017509371043`; internal mean
  `+0.234946 dB`, hard bottom-25 `+0.524294 dB`, easy top-25 `+0.041182 dB`,
  SSIM delta `+0.000093`, coverage `0.198333`.

## Locked Test Result

- Locked Haze4K test touched: yes, exactly once for the fixed policy above.
- Decision: `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`.
- Locked-test mean delta: `+0.094612 dB`.
- Hard bottom-25 delta: `+0.155218 dB`.
- Easy top-25 delta: `-0.071188 dB`.
- SSIM delta: `+0.000361`.
- Coverage: `0.164` (`164/1000` UDP accepts).
- Strong regression ratio: `0.032`.
- Worst regression ratio: `0.066`.

Failed Promotion Gate checks:

- mean delta `< +0.15 dB`;
- hard bottom-25 `< +0.30 dB`;
- easy top-25 `< -0.03 dB`;
- worst regression ratio `> 0.05`.

## Interpretation

The expert-switch direction remains mechanism- and utility-positive internally,
but this fixed A0+UDP threshold policy is not promotion-ready after locked
confirmation. Do not tune threshold, feature, checkpoint, or expert set from
the locked-test result. Any follow-up must be a new predeclared route with
selection done away from locked test.
