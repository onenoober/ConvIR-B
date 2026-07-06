# v2.33 Cross-Validation Matrix

| Hypothesis | Evidence | Verdict |
| --- | --- | --- |
| Teacher has no benefit | P1 WDMamba-alpha0.5 all `+3.2299`, hard `+4.9092`, eligible mean `+3.4569` | Rejected |
| Training path/sign broken | P2 GT `+0.0124`, positive LL `+0.0061`, sign flip `-0.0119` | Rejected as gross bug |
| S5 is catastrophic max amplification | P3 S5 `0.0750` vs decoder_pre_output `0.5148` | Rejected |
| Mask prevents tail damage | P4 severe `0`, strong-reference regression `0` | Supported |
| Mask creates useful selectivity | P4 mask effect easy/p05 negative | Rejected |
| Current carrier compresses teacher benefit | P4 masked+preservation `+0.0007` mean | Rejected |
| Need canary80 | P4 gate fail | Rejected |
