# v2.14 N1R Runtime-Valid Evidence Decision

Decision: `N1R_RUNTIME_EVIDENCE_FAIL_INSUFFICIENT_NO_TRAINING`

Locked Haze4K test touched: `false`

## Primary Gates

- benefit all-runtime ROC-AUC: `0.811898` (gate >= `0.7`)
- severe-risk all-runtime ROC-AUC: `0.826237` (gate >= `0.7`)
- benefit internal/runtime-hazy ROC-AUC: `0.802239` / `0.770909`
- severe-risk internal/runtime-hazy ROC-AUC: `0.819616` / `0.798088`
- severe-risk all-runtime minus runtime-hazy PR-AUC: `-0.014273`
- severe-risk all-runtime minus runtime-hazy top-100 enrichment: `-1.432836`
- bootstrap worse-than-margin findings: `0`

## Recommendation

Current runtime-valid evidence is insufficient; do not train this route.
