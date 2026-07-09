# CHD-RM v2e Gate Definition

- Controls must be clean: fixed permutation p-values <= 0.01 and Spearman null median <= 0.05 / p95 <= 0.10.
- D7c must beat density-only matched threshold by Spearman >= 0.15, AUROC >= 0.10, AUPRC >= 0.10, and precision >= 0.08.
- Candidate safety target: false_global <= 0.01, false_p90 <= 0.05, false_p95 <= 0.10.
- LDHN protection target: LDHN recall >= 0.10 preferred >= 0.12, with support reported.
- D2, RARM, v3, and locked Haze4K test remain forbidden.
