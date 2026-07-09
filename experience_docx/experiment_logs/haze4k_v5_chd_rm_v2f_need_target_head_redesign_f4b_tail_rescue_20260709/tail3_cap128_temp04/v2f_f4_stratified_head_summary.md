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
| f4_global_strat_control | False | 0.5380 | 0.8582 | 0.6562 | 0.2977 | 0.1515 | 0.0727 | 0 |
| f4_excess_strat_ldhn | False | 0.3143 | 0.7219 | 0.5771 | 0.2942 | 1.0000 | 0.6252 | 0 |
