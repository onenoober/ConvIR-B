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
| f4_global_strat_control | False | 0.5226 | 0.8500 | 0.6440 | 0.2973 | 0.1429 | 0.0798 | 0 |
| f4_excess_strat_ldhn | False | 0.3054 | 0.7116 | 0.5705 | 0.2963 | 1.0000 | 0.6661 | 0 |
