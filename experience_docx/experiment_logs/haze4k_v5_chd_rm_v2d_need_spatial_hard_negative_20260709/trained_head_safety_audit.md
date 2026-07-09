# v2d Trained-Head Threshold Safety Audit

| Variant | Spearman | AUROC | AUPRC | safe_global | safe_p90 | selected_threshold | coverage | recall | false_global | false_p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d7a_hn_ordinal | 0.3024 | 0.6983 | 0.4142 | 8 | 0 | 0.473467 | 0.2681 | 0.3325 | 0.0988 | 0.2680 |
| d7b_topk_hn_ordinal | 0.2950 | 0.6925 | 0.4178 | 7 | 0 | 0.477384 | 0.2529 | 0.3182 | 0.0992 | 0.2537 |
| d7s_shuffled_topk | 0.2992 | 0.6901 | 0.4309 | 24 | 0 | 0.441652 | 0.3031 | 0.3940 | 0.0526 | 0.4423 |

Thresholds are generated from train_inner predictions and evaluated on val_inner.
Locked Haze4K test usage: none.
