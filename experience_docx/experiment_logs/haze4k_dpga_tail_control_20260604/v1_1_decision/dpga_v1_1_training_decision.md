# DPGA-v1.1 Tail-Control Training Decision

Generated: 2026-06-04T13:07:27+08:00
Launch allowed: `true`

## Selected Config

- Active adapters: `shallow`
- Scale multiplier: `0.25`
- Diagnostic-only source: `true`

## Module Selection

| variant | active | score | mean | hard | easy | ssim | strong improvement | worst improvement | eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| shallow_only | `shallow` | 0.061163 | 0.026118 | 0.010513 | 0.025597 | 0.00005045 | 50.500 | 148.000 | `true` |
| no_skip | `shallow,bottleneck` | 0.052840 | 0.029635 | 0.012900 | 0.031946 | 0.00007341 | 37.000 | 103.000 | `true` |
| bottleneck_only | `bottleneck` | 0.026016 | 0.003543 | 0.002351 | 0.005628 | 0.00001915 | 47.000 | 149.000 | `false` |
| no_bottleneck | `shallow,skip` | 0.004657 | 0.022604 | 0.006640 | 0.011400 | 0.00004728 | 0.000 | 43.500 | `true` |
| all_adapters | `all` | -0.001585 | 0.025262 | 0.009149 | 0.015253 | 0.00006805 | 0.000 | 0.000 | `true` |
| skip_only | `skip` | -0.019524 | -0.000042 | -0.004025 | -0.002435 | -0.00001651 | 3.000 | 72.500 | `false` |
| no_shallow | `bottleneck,skip` | -0.024160 | 0.002592 | -0.001541 | 0.000596 | 0.00000043 | 4.000 | 34.000 | `false` |

## Scale Selection

| scale | score | mean | hard | easy | ssim | strong improvement | worst improvement | eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 0.035723 | 0.011194 | 0.002297 | 0.016642 | 0.00002077 | 37.500 | 156.500 | `true` |
| 0.00 | 0.033428 | 0.000000 | 0.000000 | -0.000000 | 0.00000000 | 106.500 | 162.000 | `false` |
| 0.50 | 0.027448 | 0.019211 | 0.004768 | 0.024835 | 0.00003951 | 18.500 | 103.000 | `true` |
| 0.75 | 0.012926 | 0.023737 | 0.007066 | 0.024283 | 0.00005516 | 6.000 | 44.000 | `true` |
| 1.00 | -0.004585 | 0.025262 | 0.009149 | 0.015253 | 0.00006806 | 0.000 | 0.000 | `true` |

## Blockers

- none

## Recommended Training Args

```json
{
  "model_name": "ConvIR-Haze4K-DPGA-v1.1-tail-control-shallow-scale0p25-seed3407-20260604",
  "dpga_active_adapters": "shallow",
  "dpga_scale_multiplier": 0.25,
  "dpga_adapter_residual_scale": 0.1,
  "dpga_tc_rec_loss": "charbonnier",
  "dpga_tc_fft_lambda": 0.05,
  "dpga_tc_anchor_lambda": 0.08,
  "dpga_tc_chroma_lambda": 0.03,
  "dpga_tc_delta_lambda": 0.0002,
  "dpga_tc_delta_tv_lambda": 5e-05,
  "dpga_tc_anchor_error_threshold": 0.035,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "stop_epoch": 20,
  "seed": 3407
}
```

Decision note: diagnostics choose the v1.1 starting configuration only; the next checkpoint must be selected on `val_inner`, not Haze4K test.
