# v2.30 Safe-Set Restricted Oracle Gap

All rows are train-derived OOF diagnostics. The restricted oracle uses target
dPSNR only after a GT-free compatibility gate has defined the safe action set.

| policy | mean | hard | easy | p05 | CVaR5 | severe | explanation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| unrestricted selected oracle | 0.790430 | 1.362880 | 0.280919 | 0.000000 | 0.000000 | 0.000000 | v2.29-style conservative selected policy |
| safe-set restricted oracle | 0.674883 | 1.237103 | 0.155230 | 0.000000 | 0.000000 | 0.000000 | best target-GT action inside GT-free compatibility gate |
| GT-free table policy | 0.014119 | 0.056478 | 0.000000 | -0.050298 | -0.305568 | 0.037500 | LCB-risk constrained fold-out table |
| no-op baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | fallback |

safe_set_to_table_mean_gap: `0.660764`
safe_set_to_table_hard_gap: `1.180625`
same_sample_oracle_mean_reference: `7.181153`
