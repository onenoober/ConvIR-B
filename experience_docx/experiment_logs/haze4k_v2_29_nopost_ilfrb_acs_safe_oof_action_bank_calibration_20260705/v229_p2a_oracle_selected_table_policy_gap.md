# v2.29 Oracle / Selected / Table-Policy Gap

All v2.29 rows are train-derived OOF diagnostics. The table-policy row uses
fold-out safety tables rather than per-sample target dPSNR preference.

| replay/policy | mean | hard | easy | p05 | CVaR5 | severe | unsafe | no-op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2.27 same-sample oracle | 7.346985 | nan | nan | 3.993753 | 3.605177 | 0.000000 | 0.000000 | n/a |
| v2.28 OOF GT-preference selected | 1.137700 | 1.376900 | 0.703500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.225 |
| v2.29 safe-envelope selected | 0.833159 | 1.375280 | 0.334240 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.2375 |
| v2.29 GT-free table policy | 0.077516 | 0.240312 | 0.037175 | -0.219939 | -0.433256 | 0.062500 | 0.000000 | n/a |

best_safe_envelope_variant: `bucket_strength_grid`
best_table_policy_variant: `energy_norm_plus_bucket_strength`
locked_test_touched: `false`
training_launched: `false`
