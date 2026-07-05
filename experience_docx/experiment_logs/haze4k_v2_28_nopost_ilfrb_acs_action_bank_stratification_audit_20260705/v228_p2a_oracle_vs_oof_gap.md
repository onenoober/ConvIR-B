# v2.28 Oracle vs OOF Gap

This table compares v2.27-style same-sample GT oracle deltas with v2.28
out-of-fold prototype replay. All rows are train-derived; locked test is untouched.

| replay type | mean | hard | easy | p05 | unsafe | no-op pref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same-sample oracle | 7.346968 | 8.332439 | 6.044410 | 3.993674 | 0.000000 | n/a |
| OOF prototype selected | 1.137656 | 1.376888 | 0.703518 | 0.000000 | 0.000000 | 0.225 |
| cross-bucket | -0.132569 | nan | nan | -2.461589 | 0.455556 | n/a |
| negative controls | -0.429817 | nan | nan | -4.061279 | 0.550385 | n/a |

sample_count: `80`
stage_sets: `S6_early_mid_final,S5_bottleneck_mid,S4_final_decoder`
prototype_aggregate: `median`
locked_test_touched: `false`
