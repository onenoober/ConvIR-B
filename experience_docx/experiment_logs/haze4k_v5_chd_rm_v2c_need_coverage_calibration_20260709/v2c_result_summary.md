# CHD-RM v2c Need Coverage Calibration Summary

Decision: `PAUSE_V2C_SCALE_CALIBRATION_NOT_ENOUGH`

| Variant | Method | Pearson | Spearman | AUROC | Coverage | False-strong | Monotonic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d6a_quantile | identity | 0.2710 | 0.2465 | 0.6611 | 0.0000 | 0.0000 | 3/4 |
| d6a_quantile | affine_p01_p99 | 0.2706 | 0.2465 | 0.6611 | 0.3568 | 0.1301 | 3/4 |
| d6a_quantile | affine_p10_p90 | 0.2593 | 0.2455 | 0.6606 | 0.3653 | 0.1340 | 3/4 |
| d6a_quantile | mean_std | 0.2571 | 0.2456 | 0.6607 | 0.3098 | 0.1099 | 3/4 |
| d6a_quantile | logit_mean_std | 0.2636 | 0.2465 | 0.6611 | 0.4209 | 0.1625 | 3/4 |
| d6a_quantile | quantile_map_1001 | 0.2436 | 0.2465 | 0.6611 | 0.3379 | 0.1217 | 3/4 |
| d6b_log | identity | 0.2712 | 0.2914 | 0.6883 | 0.0000 | 0.0000 | 2/2 |
| d6b_log | affine_p01_p99 | 0.2709 | 0.2914 | 0.6883 | 0.2270 | 0.0932 | 2/2 |
| d6b_log | affine_p10_p90 | 0.2607 | 0.2907 | 0.6881 | 0.0929 | 0.0328 | 2/2 |
| d6b_log | mean_std | 0.2513 | 0.2893 | 0.6875 | 0.0691 | 0.0210 | 2/2 |
| d6b_log | logit_mean_std | 0.2392 | 0.2914 | 0.6883 | 0.1582 | 0.0630 | 2/2 |
| d6b_log | quantile_map_1001 | 0.2142 | 0.2914 | 0.6883 | 0.0960 | 0.0346 | 2/2 |
| d6c_ordinal_quantile | identity | 0.3632 | 0.3298 | 0.7133 | 0.0000 | 0.0000 | 4/4 |
| d6c_ordinal_quantile | affine_p01_p99 | 0.3630 | 0.3298 | 0.7133 | 0.4199 | 0.1683 | 4/4 |
| d6c_ordinal_quantile | affine_p10_p90 | 0.3504 | 0.3288 | 0.7130 | 0.3824 | 0.1486 | 4/4 |
| d6c_ordinal_quantile | mean_std | 0.3490 | 0.3290 | 0.7131 | 0.3159 | 0.1153 | 4/4 |
| d6c_ordinal_quantile | logit_mean_std | 0.3495 | 0.3298 | 0.7133 | 0.4093 | 0.1623 | 4/4 |
| d6c_ordinal_quantile | quantile_map_1001 | 0.3270 | 0.3298 | 0.7133 | 0.3448 | 0.1287 | 4/4 |
| d6s_shuffled_quantile | identity | -0.2487 | -0.2530 | 0.3369 | 0.0000 | 0.0000 | 0/4 |
| d6s_shuffled_quantile | affine_p01_p99 | -0.2508 | -0.2530 | 0.3369 | 0.2485 | 0.4784 | 0/4 |
| d6s_shuffled_quantile | affine_p10_p90 | -0.2525 | -0.2529 | 0.3369 | 0.3182 | 0.5524 | 0/4 |
| d6s_shuffled_quantile | mean_std | -0.2519 | -0.2529 | 0.3369 | 0.2922 | 0.5249 | 0/4 |
| d6s_shuffled_quantile | logit_mean_std | -0.2525 | -0.2530 | 0.3369 | 0.3724 | 0.6030 | 0/4 |
| d6s_shuffled_quantile | quantile_map_1001 | -0.2513 | -0.2530 | 0.3369 | 0.3233 | 0.5575 | 0/4 |

Calibration is fitted on train_inner predictions only and evaluated on val_inner.
Forbidden in this stage: D2, RARM connection, v3 expansion, locked Haze4K test.
Locked Haze4K test usage: none.
