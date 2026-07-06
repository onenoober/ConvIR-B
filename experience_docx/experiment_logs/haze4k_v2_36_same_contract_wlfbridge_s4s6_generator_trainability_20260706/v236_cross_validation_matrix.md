# v2.36 Cross-Validation Matrix

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| Full-image same-context WDMamba has average benefit | mean +3.2299, hard +4.9092, easy +1.1266 | Supported |
| Unmasked alpha0.5 substrate is tail-safe | CVaR5 -0.7438, severe 0.035, strong-reference regression 0.1733 | Rejected |
| Failure is infra/command issue | Postrun audit recomputed 600 rows, 30 negatives, 21 severe | Rejected |
| Bridge/generator failed | P1/P2 not authorized or run | Not tested |
| S4+S6 full600 masked representability failed | Masked projection not run | Not tested |
| Next route can launch bridge | P0 gate failed | Rejected |
| Next route can audit tail-safe mask/no-op contract | Family summary permits a new same-context route with full600 tail-safe contract first | Supported |
