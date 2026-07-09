# CHD-RM v2d Need Spatial Hard-Negative Summary

Decision: `PAUSE_V2D_D7A_D7B_NOT_ENOUGH_NEXT_D7C_OR_TARGET_AUDIT`

| Variant | Pearson | Spearman | AUROC | AUPRC | Coverage | False global | False p90 | Recall | Mono | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d7a_hn_ordinal | 0.3485 | 0.3024 | 0.6983 | 0.4142 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/4 | FAIL |
| d7b_topk_hn_ordinal | 0.3399 | 0.2950 | 0.6925 | 0.4178 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/4 | FAIL |
| d7s_shuffled_topk | 0.3072 | 0.2992 | 0.6901 | 0.4309 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4/4 | FAIL |
| d7s2_density_only_control | 0.2750 | 0.2958 | 0.6890 | 0.4336 | 0.1045 | 0.0037 | 0.0051 | 0.1467 | 4/4 | FAIL |

Gate: Pearson >= 0.33, Spearman >= 0.32, AUROC >= 0.70, AUPRC >= 0.42, monotonic 4/4, coverage in [0.20, 0.40], false global <= 0.10, false p90 <= 0.15, recall >= 0.25.

Locked Haze4K test usage: none.
D2/RARM/v3 expansion: forbidden in this stage.
