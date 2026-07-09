# CHD-RM v2e Final Closeout

Decision: `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

Summary:

- E0/E1/E2 controls are clean enough to keep D7c top-k as a real `R_need` candidate: fixed image permutation passes, and D7c beats density-only matched threshold.
- E3 fails LDHN protection: frozen D7c top-k LDHN recall is `0.0370` at safe false-tail.
- D7c-RP was authorized inside v2e and run with 5 recall-protection strengths. It raises LDHN recall, but every LDHN-passing point violates false-tail safety.
- Therefore v3 no-op RARM, RARM training, D2, and locked-test use remain blocked.

Key main-audit metrics:

| item | value |
| --- | ---: |
| D7c top-k Spearman | 0.5175 |
| D7c top-k AUROC | 0.8456 |
| D7c top-k AUPRC | 0.6442 |
| D7c top-k coverage | 0.3027 |
| D7c top-k false-global | 0.0030 |
| D7c top-k false-p90 | 0.0246 |
| D7c top-k false-p95 | 0.0476 |
| D7c top-k LDHN recall | 0.0370 |
| fixed permutation Spearman p | 0.0099 |
| density gap Spearman | 0.2178 |
| density gap AUROC | 0.1481 |
| density gap AUPRC | 0.1046 |

D7c-RP full audit:

| variant | pass | Spearman | AUROC | AUPRC | coverage | false-p90 | false-p95 | LDHN recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d7c_rp_lam05_r3_lr2e4 | False | 0.5327 | 0.8544 | 0.6526 | 0.2956 | 0.0326 | 0.0741 | 0.0491 |
| d7c_rp_lam10_r3_lr2e4 | False | 0.5284 | 0.8517 | 0.6512 | 0.2974 | 0.0599 | 0.2069 | 0.1096 |
| d7c_rp_lam15_r3_lr15e4 | False | 0.5276 | 0.8477 | 0.6531 | 0.2989 | 0.0714 | 0.2941 | 0.1428 |
| d7c_rp_lam20_r5_lr15e4 | False | 0.5116 | 0.8407 | 0.6465 | 0.3013 | 0.1081 | 0.3545 | 0.1667 |
| d7c_rp_lam30_r5_lr1e4 | False | 0.5149 | 0.8417 | 0.6481 | 0.3024 | 0.1684 | 0.5348 | 0.1822 |

Interpretation:

The RP sweep shows a real safety/recall tradeoff. The weakest RP point remains safe-ish but has LDHN recall `0.0491` below the `0.10` line. The first LDHN-passing point (`d7c_rp_lam10_r3_lr2e4`) reaches LDHN recall `0.1096`, but false-p90/false-p95 rise to `0.0599` / `0.2069`. Stronger RP points increase LDHN recall further while worsening false-tail.

Locked Haze4K test usage: none.
D2/RARM/v3 runtime: not run.
