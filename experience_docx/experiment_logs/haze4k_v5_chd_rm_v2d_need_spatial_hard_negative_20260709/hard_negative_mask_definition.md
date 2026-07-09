# Hard-Negative Mask Definition

`low_context_hard_negative = need_low AND density_low`.
`need_low` is the quantile R_need target <= q33. `density_low` is the normalized density target <= q33.
Both thresholds are train_inner-derived. Low density alone is not penalized.
