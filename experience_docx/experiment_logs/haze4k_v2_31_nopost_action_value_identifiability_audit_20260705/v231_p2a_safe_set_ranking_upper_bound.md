# v2.31 Safe-Set Ranking Upper Bound

All models are nested fold-out diagnostics using train-derived rows only.
Features exclude target GT and locked-test data. Labels are safe-set action
dPSNR for route diagnosis, not deployment input.

safe_set_oracle_mean: `0.674882`
safe_set_oracle_hard: `1.237125`
v2.30_table_mean: `0.014128`

| model | top1 | top3 | NDCG@3 | mean | hard | easy | p05 | CVaR5 | severe | gap | real-vs-shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_ranker | 0.300000 | 0.487500 | 0.750367 | 0.387372 | 0.673985 | 0.096140 | -0.691904 | -1.335754 | 0.225000 | 0.287509 | 0.061315 |
| shallow_tree_median_bins | 0.087500 | 0.150000 | 0.446891 | 0.136188 | 0.121394 | 0.019826 | -0.142586 | -0.341225 | 0.037500 | 0.538694 | -0.189870 |
| kNN_nonparametric | 0.287500 | 0.425000 | 0.705340 | 0.423387 | 0.815245 | 0.099477 | -0.442120 | -1.258163 | 0.137500 | 0.251495 | 0.097329 |
| bucket_only_baseline | 0.087500 | 0.275000 | 0.712655 | 0.319625 | 0.512849 | 0.085579 | -0.458754 | -0.750183 | 0.200000 | 0.355256 | -0.006432 |
| shuffled_label_control | 0.100000 | 0.237500 | 0.633653 | 0.326058 | 0.655687 | 0.079441 | -0.530249 | -0.921622 | 0.150000 | 0.348824 | 0.000000 |

ranking_gate_pass: `False`
best_model: `kNN_nonparametric`
