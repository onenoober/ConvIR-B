# Haze4K v1.7 Risk-Controlled Expert Mix Analysis

Status: read `v17_analysis_status.json`.

Primary files:

- `v17_oracle_switch_mix_alpha_grid.json`: fixed alpha and GT-oracle alpha
  upper-bound summaries.
- `v17_oof_gain_risk_predictability.csv`: OOF gain/risk head ROC/PR/Brier
  diagnostics for alpha `1.0/0.75/0.5/0.25`.
- `v17_oof_risk_coverage_curves.csv`: searched risk-controlled alpha policy
  curves over gain, risk, and OOD thresholds.
- `v17_policy_stability_by_fold.csv`: selected policy metrics by OOF fold.
- `v17_calibration_curve_bad_risk.csv`: risk probability calibration bins.
- `v17_trainheldout_confirm_summary.json`: train_inner fit/OOF selection and
  val_regular+val_hard holdout confirmation.
- `v17_oof_policy_per_image.csv` and `v17_trainheldout_policy_per_image.csv`:
  selected alpha and per-image deltas.

Locked Haze4K test touched: no.

Interpretation: oracle files are upper bounds. Only OOF plus train-derived
heldout confirmation can authorize a later immutable locked-test command.
