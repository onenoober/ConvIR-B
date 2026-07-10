# v2f F4b Tail-Rescue Matrix Summary

Status: `COMPLETED_GATE_FAIL`

Policy: ConvIR-B and D3 frozen; D2/v3/RARM not run; locked Haze4K test not used.

| Spec | Variant | Gate | Spearman | AUROC | AUPRC | Coverage | False p95 | LDHN recall | Safe+LDHN points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tail2_topk10 | f4_global_strat_control | False | 0.5226 | 0.8500 | 0.6440 | 0.2973 | 0.1429 | 0.0798 | 0 |
| tail2_topk10 | f4_excess_strat_ldhn | False | 0.3054 | 0.7116 | 0.5705 | 0.2963 | 1.0000 | 0.6661 | 0 |
| tail3_cap128_temp04 | f4_global_strat_control | False | 0.5380 | 0.8582 | 0.6562 | 0.2977 | 0.1515 | 0.0727 | 0 |
| tail3_cap128_temp04 | f4_excess_strat_ldhn | False | 0.3143 | 0.7219 | 0.5771 | 0.2942 | 1.0000 | 0.6252 | 0 |
| tail4_topk20 | f4_global_strat_control | False | 0.5179 | 0.8477 | 0.6418 | 0.2986 | 0.0246 | 0.0262 | 0 |
| tail4_topk20 | f4_excess_strat_ldhn | False | 0.3392 | 0.7344 | 0.5847 | 0.2999 | 0.9895 | 0.5368 | 0 |

Decision: keep the original v2e global safety/LDHN gate as the primary contract.