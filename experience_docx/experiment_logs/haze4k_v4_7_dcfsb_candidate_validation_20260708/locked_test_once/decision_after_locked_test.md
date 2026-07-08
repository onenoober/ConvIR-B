# Haze4K v4.7 Locked-Test Confirmation Decision

Status: `LOCKED_TEST_CONFIRM_FAIL`

Route id: `haze4k_v4_7_dcfsb_candidate_validation_20260708`

Fixed candidate: v4.6 DCFSB-bottleneck `adapter4` checkpoint.

Policy note: a prior post-v4.7 directory-count preflight enumerated the locked test split but produced no metric. This decision records the sole metric-producing locked-test command for the fixed candidate.

Prediction images saved: `false`

Metric-producing locked-test command count: `1`

## Metrics

| Metric | Value |
| --- | ---: |
| A0 mean PSNR | `34.145502` |
| Candidate mean PSNR | `34.149328` |
| mean dPSNR | `0.003826` |
| median dPSNR | `-0.003686` |
| p5 dPSNR | `-0.210819` |
| p95 dPSNR | `0.182354` |
| positive ratio | `0.484000` |
| A0 mean SSIM | `0.989726` |
| Candidate mean SSIM | `0.989747` |
| mean dSSIM | `0.00002084` |

## Gate Results

```json
{
  "mean_delta_psnr": true,
  "mean_delta_ssim": true,
  "metric_command_count_one": true,
  "p5_delta_psnr": true,
  "positive_ratio": false,
  "save_prediction_images_false": true
}
```

## Decision

The locked-test confirmation gate failed. Do not run additional locked-test commands for this candidate. Do not tune from locked-test results.
