# CHD-RM v2e D7c-RP Micro-Variant Summary

Decision: `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

Best RP variant by gate ordering: `d7c_rp_lam30_r5_lr1e4`.

Frozen and forbidden: ConvIR-B frozen, D3 density frozen, no D2, no RARM connection/training, no v3 runtime, no locked Haze4K test.

## Baseline

D7c top-k baseline Spearman `0.5175`, AUROC `0.8456`, AUPRC `0.6442`, coverage `0.2971`, false-p90 `0.0222`, false-p95 `0.0430`, LDHN recall `0.0343`.

## RP Variants

| variant | pass | Spearman | AUROC | AUPRC | coverage | recall | precision | false_global | false_p90 | false_p95 | LDHN recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d7c_rp_lam05_r3_lr2e4 | False | 0.5327 | 0.8544 | 0.6526 | 0.2956 | 0.4429 | 0.6371 | 0.0036 | 0.0326 | 0.0741 | 0.0491 |
| d7c_rp_lam10_r3_lr2e4 | False | 0.5284 | 0.8517 | 0.6512 | 0.2974 | 0.4498 | 0.6433 | 0.0107 | 0.0599 | 0.2069 | 0.1096 |
| d7c_rp_lam15_r3_lr15e4 | False | 0.5276 | 0.8477 | 0.6531 | 0.2989 | 0.4561 | 0.6490 | 0.0143 | 0.0714 | 0.2941 | 0.1428 |
| d7c_rp_lam20_r5_lr15e4 | False | 0.5116 | 0.8407 | 0.6465 | 0.3013 | 0.4559 | 0.6434 | 0.0204 | 0.1081 | 0.3545 | 0.1667 |
| d7c_rp_lam30_r5_lr1e4 | False | 0.5149 | 0.8417 | 0.6481 | 0.3024 | 0.4594 | 0.6461 | 0.0241 | 0.1684 | 0.5348 | 0.1822 |

## Train LDHN Oversampling Support

Train LDHN p75 cutoff: `0.122975`.
Images at or above cutoff: `601` / `2400`.
