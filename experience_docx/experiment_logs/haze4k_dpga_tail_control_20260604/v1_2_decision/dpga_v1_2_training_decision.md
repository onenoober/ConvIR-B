# DPGA-v1.2 Training Decision

Generated: 2026-06-04T13:53:22+08:00
Launch allowed: `true`

## Checks

| check | observed | required | pass |
| --- | ---: | --- | --- |
| previous_gate_failed | False | false | `true` |
| locked_test_blocked | False | false | `true` |
| mean_gain_ok | 0.037036 | >= +0.030 dB | `true` |
| hard_gain_shortfall | 0.023367 | < +0.030 dB | `true` |
| tail_regressions_safe | 9 | <= 12 / 300 | `true` |
| strong_regressions_safe | 12 | <= 15 / 75 | `true` |

## Selected Config

```json
{
  "model_name": "ConvIR-Haze4K-DPGA-v1.2-hard-gain-shallow-scale0p5-anchor0p04-seed3407-20260604",
  "dpga_active_adapters": "shallow",
  "dpga_scale_multiplier": 0.5,
  "dpga_adapter_residual_scale": 0.1,
  "dpga_tc_rec_loss": "charbonnier",
  "dpga_tc_fft_lambda": 0.05,
  "dpga_tc_anchor_lambda": 0.04,
  "dpga_tc_chroma_lambda": 0.03,
  "dpga_tc_delta_lambda": 0.00025,
  "dpga_tc_delta_tv_lambda": 5e-05,
  "dpga_tc_anchor_error_threshold": 0.035,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "stop_epoch": 20,
  "seed": 3407
}
```

## Rationale

v1.1 passed mean/positive/tail safety on val_inner but missed hard-bottom gain. v1.2 keeps the same adapter family, raises scale from 0.25 to 0.5, and lowers anchor pressure from 0.08 to 0.04 to test whether hard gain can cross the gate without reviving tail regressions.

Locked test remains blocked until this config passes the same `val_inner` gate.
