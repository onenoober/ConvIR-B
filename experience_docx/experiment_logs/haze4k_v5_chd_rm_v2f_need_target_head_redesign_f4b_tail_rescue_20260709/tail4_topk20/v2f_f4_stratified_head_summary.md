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
| f4_global_strat_control | False | 0.5179 | 0.8477 | 0.6418 | 0.2986 | 0.0246 | 0.0262 | 0 |
| f4_excess_strat_ldhn | False | 0.3392 | 0.7344 | 0.5847 | 0.2999 | 0.9895 | 0.5368 | 0 |
