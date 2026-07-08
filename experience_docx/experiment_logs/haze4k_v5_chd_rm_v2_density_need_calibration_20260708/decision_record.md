# CHD-RM v2 Decision Record

Decision: `PAUSE_V2_DUAL_HEAD_NOT_PASSED`

Gate contract:

- density Pearson >= 0.25, Spearman >= 0.30, AUROC >= 0.65, 4/4 monotonic pairs.
- need Pearson >= 0.20, Spearman >= 0.25, AUROC >= 0.65, 4/4 monotonic pairs.
- density and need strong-response coverage must be non-degenerate: 0.01 to 0.90.
- low-haze false-strong-recovery rate <= 0.10.
- shuffled target control must not pass the same gate.

Next step: Inspect D3/D4 single-head evidence; run D2 only if single-head learnability is clear.

Locked Haze4K test usage: none.
