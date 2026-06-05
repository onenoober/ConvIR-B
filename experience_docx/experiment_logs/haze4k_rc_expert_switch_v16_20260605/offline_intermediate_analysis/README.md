# Haze4K v1.6 Risk-Calibrated Expert Switch Evidence

Status: `OFFLINE_INTERMEDIATE_ANALYSIS_COMPLETE` after the analysis script finishes.

Primary files:

- `route_utility_leaderboard.csv`: unified retrospective Mechanism/Utility/Promotion gate table.
- `hard_expert_leaderboard.csv`: hard expert candidate ranking.
- `global_model_leaderboard.csv`: strict promotion-style ranking.
- `utility_tradeoff_leaderboard.csv`: risk-adjusted utility ranking.
- `expert_bank_oracle_switch_a0_udp.json`: GT oracle upper bound for A0 + official UDPNet.
- `expert_bank_oracle_switch_a0_udp_per_image.csv`: oracle per-image switch decisions.
- `udp_accept_label_predictability_oof.csv`: first accept-label predictability diagnostic.
- `udp_bad_risk_predictability_oof.csv`: first bad-risk predictability diagnostic.
- `udp_accept_label_training_table.csv`: label table for later deployable router work.
- `rc_expert_switch_oof_summary.json`: first threshold-switch OOF diagnostic.
- `rc_expert_switch_oof_policy_search.csv`: searched threshold policies.

Locked Haze4K test touched: no.

Important interpretation: oracle and proxy-threshold switch outputs are
intermediate diagnostics. Promotion still requires a deployable feature
audit and OOF or held-out calibration that does not select thresholds on
the locked test.

## Oracle Snapshot

- Status: `completed_offline_gt_oracle`
- UDP accept ratio: `0.53`
- Oracle gate pass: `True`

## Switch Diagnostic Snapshot

- Status: `completed_true_oof_threshold_search`
- Utility-gate policies: `None`
- Promotion-gate policies: `None`
