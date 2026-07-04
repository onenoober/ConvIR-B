# v2.20 P1 O3 Context Learnability Decision

Decision: `P1A_PASS_MECHANISM_P1B_FAIL_CONTINUE_DIAGNOSTICS_NO_TRAINING`

- primary predictor: `P1_final_mid_global_context_predictor`
- mean dPSNR: `2.068434411684672`
- hard bottom25 dPSNR: `4.1449682362874345`
- easy top25 dPSNR: `0.5198832130432129`
- positive ratio: `0.8508333333333333`
- p05 dPSNR: `-0.7255226135253906`
- CVaR5 dPSNR: `-1.6966981569925943`
- severe rate: `0.11125`
- strong-reference regression rate: `0.26666666666666666`
- wrong-direction rate: `0.004166666666666667`
- control gap vs shuffled: `3.1958893815676372`
- fold tail pass count: `0` / 5

Mechanism gate checks:

{
  "hard_bottom25_ge_0p50": true,
  "mean_dPSNR_ge_0p25": true,
  "positive_ratio_ge_0p60": true,
  "real_beats_shuffled_by_0p20": true,
  "target_mse_beats_shuffled_and_final_only": true,
  "wrong_direction_rate_le_0p12": true
}

Training-authorization safety gate checks:

{
  "CVaR5_ge_neg0p35": false,
  "easy_top25_ge_neg0p02": true,
  "fold_tail_pass_ge_4_of_5": false,
  "p05_ge_neg0p15": false,
  "severe_rate_le_0p035": false,
  "strong_easy_mean_ge_0": true,
  "strong_easy_p05_ge_neg0p15": false,
  "strong_reference_regression_rate_le_0p075": false
}

P1 does not launch training. P1-B pass would only authorize a separate route-card review for N3 microfit.
Locked Haze4K remains untouched.
