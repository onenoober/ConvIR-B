# v2.31 Local Optimum Audit

Question: does adding target-only physics/frequency evidence rescue action-value
selection, or does the v2.30 bank remain a no-op local optimum?

feature_gate_pass: `False`
ranking_gate_pass: `False`
best_ranker: `kNN_nonparametric`
real_policy_mean: `0.42338697910308837`
shuffled_label_mean: `0.3260577440261841`
best_coverage_hard_gain: `0.8152445793151856` at coverage `0.6`

Interpretation rule: if separability and nested ranking both fail, the
current discrete action-bank selector route should close rather than receive
more table/firewall micro-tuning.
