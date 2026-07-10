# v2f F4 Stratified Head Canary Summary

Status: `COMPLETED_GATE_FAIL`

Policy:

- ConvIR-B frozen: yes
- D3 density frozen: yes
- D2/v3/RARM: not run
- Locked Haze4K test usage: none

Original v2e gate is still the primary decision contract. The density-conditioned target is a training redesign, not a replacement for the safety/LDHN audit.

| Variant | Gate | Spearman | AUROC | AUPRC | Coverage | False p95 | LDHN recall | Safe+LDHN points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f4_global_strat_control | False | 0.5168 | 0.8451 | 0.6454 | 0.2994 | 0.7500 | 0.2634 | 0 |
| f4_cond_strat_core | False | 0.2724 | 0.6900 | 0.5516 | 0.2940 | 1.0000 | 0.7104 | 0 |
| f4_cond_strat_ldhn | False | 0.2526 | 0.6798 | 0.5461 | 0.2923 | 1.0000 | 0.7343 | 0 |
| f4_excess_strat_ldhn | False | 0.3236 | 0.7211 | 0.5725 | 0.2944 | 1.0000 | 0.7005 | 0 |
