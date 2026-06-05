# Haze4K v1.7 Risk-Controlled Expert Mix Evidence

Route card:
`experience_docx/experiment_cards/2026-06-05-haze4k-convir-v1-7-rc-expert-mix.md`.

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`.

Primary files:

- `v17_fulltrain_features/v17_fulltrain_a0_udp_feature_table.csv`
- `v17_fulltrain_features/v17_fulltrain_a0_udp_feature_summary.json`
- `v17_mix_analysis/v17_oracle_switch_mix_alpha_grid.json`
- `v17_mix_analysis/v17_oof_gain_risk_predictability.csv`
- `v17_mix_analysis/v17_oof_risk_coverage_curves.csv`
- `v17_mix_analysis/v17_policy_stability_by_fold.csv`
- `v17_mix_analysis/v17_calibration_curve_bad_risk.csv`
- `v17_mix_analysis/v17_trainheldout_confirm_summary.json`
- `v17_mix_analysis/v17_analysis_status.json`
- `v17_mix_analysis_orderfix_rerun.log`

Launcher:
`run_v17_intermediate_analysis.sh`.

Locked Haze4K test touched: no. This evidence root is for train-derived
feature extraction and policy calibration only.

## Key Metrics

Full-train feature extraction completed with `3000` rows across
`train_inner`, `val_regular`, and `val_hard`.

GT oracle alpha mix: mean `+0.8689 dB`, hard bottom-25 `+0.9623 dB`, easy
top-25 `+0.8245 dB`, SSIM `+0.000283`, worst ratio `0`, strong ratio `0`.
This passes mechanism/oracle gate only.

Selected full-train OOF policy: coverage `0.1557`, mean `+0.1079 dB`, hard
bottom-25 `+0.1417 dB`, easy top-25 `+0.1020 dB`, SSIM `+0.000054`, worst
ratio `0.0067`, strong ratio `0.0107`, fold utility pass count `0/5`. OOF gate
failed on mean and hard-gain margins.

Train-derived heldout confirmation: coverage `0.1117`, mean `+0.0945 dB`, hard
bottom-25 `+0.1297 dB`, easy top-25 `+0.0597 dB`, SSIM `+0.000041`, worst
ratio `0.0033`, strong ratio `0.0282`. Heldout gate failed on mean and
hard-gain margins.

Decision: keep `v17_fulltrain_a0_udp_feature_table.csv` and the alpha-grid
analysis as reusable calibration assets, but do not touch locked Haze4K test
from the current v1.7A low-capacity risk-control policy.
