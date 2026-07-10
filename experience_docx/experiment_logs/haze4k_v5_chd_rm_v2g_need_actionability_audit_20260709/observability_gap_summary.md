# v2g Observability Gap Summary

Current deployable frozen/current features show LDHN-vs-LDLN separability (best AUROC 0.8107, AUPRC 0.8078), but F4/F4b could not turn this into a safe LDHN operating point.

The key semantic gap is that global LDHN is mostly isolated from haze adjacency: isolated fraction 0.8907, adjacent-to-haze fraction 0.1093. D7c recall is higher on haze-adjacent LDHN (0.1559) than isolated LDHN (0.0224).

Interpretation: future work should define actionable LDHN before training another head.
