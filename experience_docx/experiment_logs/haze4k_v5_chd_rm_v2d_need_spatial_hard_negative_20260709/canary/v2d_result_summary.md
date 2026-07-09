# CHD-RM v2d Need Spatial Hard-Negative Summary

Decision: `PAUSE_V2D_D7A_D7B_NOT_ENOUGH_NEXT_D7C_OR_TARGET_AUDIT`

| Variant | Pearson | Spearman | AUROC | AUPRC | Coverage | False global | False p90 | Recall | Mono | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d7a_hn_ordinal | 0.3423 | 0.3221 | 0.7121 | 0.4048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/4 | FAIL |
| d7b_topk_hn_ordinal | 0.3065 | 0.2802 | 0.6861 | 0.3881 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3/4 | FAIL |
| d7s_shuffled_topk | 0.3287 | 0.3233 | 0.7141 | 0.4367 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/4 | FAIL |
| d7s2_density_only_control | 0.3161 | 0.3368 | 0.7198 | 0.4279 | 0.1229 | 0.0045 | 0.0069 | 0.1861 | 4/4 | FAIL |

Gate: Pearson >= 0.33, Spearman >= 0.32, AUROC >= 0.70, AUPRC >= 0.42, monotonic 4/4, coverage in [0.20, 0.40], false global <= 0.10, false p90 <= 0.15, recall >= 0.25.

Locked Haze4K test usage: none.
D2/RARM/v3 expansion: forbidden in this stage.
