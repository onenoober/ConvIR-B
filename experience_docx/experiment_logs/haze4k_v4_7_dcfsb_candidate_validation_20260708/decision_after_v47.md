# Decision After Haze4K v4.7

Status: `PASS`

Route id: `haze4k_v4_7_dcfsb_candidate_validation_20260708`

Candidate: fixed v4.6 DCFSB-bottleneck `adapter4` checkpoint.

Locked test: not touched; not enumerated.

## Metrics

| Metric | Value | Gate |
| --- | ---: | --- |
| mean dPSNR | `0.044404` | `>= +0.035` |
| positive ratio | `0.625000` | `>= 0.58` |
| p5 dPSNR | `-0.216141` | `>= -0.25` |
| mean dHighL1 | `-0.0000005702` | `<= +0.000005` |
| bootstrap CI low | `0.024481` | `> 0` |
| sign-test p | `3.802649e-05` | `< 0.01` |
| systematic failure flags | `0` | `0` |

## Gate Results

```json
{
  "bootstrap_ci95_low": true,
  "locked_test_not_touched": true,
  "mean_delta_high_l1": true,
  "mean_delta_psnr": true,
  "no_systematic_failure_bin": true,
  "p5_delta_psnr": true,
  "positive_ratio": true,
  "sign_test": true,
  "test_split_not_enumerated": true
}
```

## Decision

Adapter4 is eligible for a separate written confirmation gate for exactly one fixed-checkpoint locked-test command. Do not run locked test from this v4.7 script.
