# v2.21 P1 Safety-Gated Replay Decision

Decision: `V221_P1_REPLAY_GATE_PASS_REVIEW_N3_MICROFIT_ROUTE_CARD_NO_TRAINING_LAUNCHED`

- selected fixed OOF candidate: `V221_risk_temperature_gamma0p50`
- selected gate kind: `predicted_risk_temperature`
- selected threshold: ``
- mean dPSNR: `2.2270253642400104`
- hard bottom25 dPSNR: `4.303077551523844`
- easy top25 dPSNR: `0.7403017361958821`
- p05 dPSNR: `-0.0025074005126953123`
- CVaR5 dPSNR: `-0.20886227289835613`
- severe rate: `0.017916666666666668`
- strong-reference regression rate: `0.04833333333333333`
- fold tail pass count: `5` / 5
- selected candidate gate pass: `True`
- factorial A gate pass: `True`

Gate checks:

{
  "CVaR5_ge_neg0p35": true,
  "calibration_gated_high_prob_severe_rate": 0.020134228187919462,
  "calibration_high_prob_count": 298,
  "calibration_high_prob_severe_reduction": 0.5302013422818792,
  "calibration_noop_bin_count": 298,
  "calibration_noop_bin_mean_dPSNR": 0.23659139671581703,
  "calibration_raw_high_prob_severe_rate": 0.5503355704697986,
  "easy_top25_ge_0p00": true,
  "failed_check_count": 0,
  "fold_tail_pass_count": 5,
  "fold_tail_pass_ge_4_of_5": true,
  "gate_kind": "predicted_risk_temperature",
  "hard_bottom25_ge_2p00": true,
  "high_prob_severe_rate_clearly_reduced": true,
  "mean_dPSNR_ge_1p00": true,
  "noop_bin_mean_ge_neg0p03": true,
  "p05_ge_neg0p15": true,
  "positive_ratio_ge_0p75": true,
  "severe_rate_le_0p035": true,
  "strong_easy_p05_ge_neg0p15": true,
  "strong_reference_regression_rate_le_0p075": true,
  "threshold": "",
  "training_authorization_pass": true,
  "variant": "V221_risk_temperature_gamma0p50"
}

Factorial A checks:

{
  "CVaR5_ge_neg0p35": true,
  "calibration_gated_high_prob_severe_rate": 0.020134228187919462,
  "calibration_high_prob_count": 298,
  "calibration_high_prob_severe_reduction": 0.5302013422818792,
  "calibration_noop_bin_count": 298,
  "calibration_noop_bin_mean_dPSNR": 0.23659139671581703,
  "calibration_raw_high_prob_severe_rate": 0.5503355704697986,
  "easy_top25_ge_0p00": true,
  "failed_check_count": 0,
  "fold_tail_pass_count": 5,
  "fold_tail_pass_ge_4_of_5": true,
  "gate_kind": "factor_A_pred_action_pred_gate",
  "hard_bottom25_ge_2p00": true,
  "high_prob_severe_rate_clearly_reduced": true,
  "mean_dPSNR_ge_1p00": true,
  "noop_bin_mean_ge_neg0p03": true,
  "p05_ge_neg0p15": true,
  "positive_ratio_ge_0p75": true,
  "severe_rate_le_0p035": true,
  "strong_easy_p05_ge_neg0p15": true,
  "strong_reference_regression_rate_le_0p075": true,
  "threshold": "",
  "training_authorization_pass": true,
  "variant": "V221_factor_A_pred_action_pred_gate"
}

No training, no N3 microfit, and no locked-test command is launched by v2.21.
