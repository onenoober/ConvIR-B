# v2.19 P1 O2 Spatial Action Learnability Decision

Decision: `P1_FAIL_O2_SPATIAL_ACTION_NOT_SAFELY_LEARNED`

- primary spatial predictor: `P1_small_cnn_spatial`
- mean dPSNR: `0.9921334060033162`
- hard bottom25 dPSNR: `2.6504010645548504`
- easy top25 dPSNR: `-0.1345879618326823`
- positive ratio: `0.7191666666666666`
- p05 dPSNR: `-1.148581314086914`
- CVaR5 dPSNR: `-2.10583610534668`
- severe rate: `0.2025`
- strong-reference regressions: `302` / `600`
- wrong-direction rate: `0.03625`
- control gap vs shuffled: `1.8397358504931132`
- fold tail pass count: `0` / 5

Gate checks:

{
  "CVaR5_ge_neg0p25": false,
  "beats_global_broadcast_tail": false,
  "easy_top25_ge_0": false,
  "fold_tail_4_of_5": false,
  "hard_bottom25_ge_0p50": true,
  "mean_dPSNR_ge_0p30": true,
  "p05_ge_neg0p10": false,
  "positive_ratio_ge_0p65": true,
  "real_beats_shuffled_by_0p20": true,
  "severe_rate_le_0p025": false,
  "strong_regressions_le_5pct": false,
  "wrong_direction_rate_le_0p10": true
}

Interpretation:

- P1 is a deployable-policy learnability audit, not WLDB-B training.
- If P1 fails, training remains blocked and P2/P3 risk/objective diagnostics provide the closeout reason.
- Locked Haze4K remains untouched.
