# v4.8 R-only Calibration Decision

Probe: `v48_r_only_calibration_sdc_lite_oof3000`

Decision: **FAIL**.

Scope: train-derived OOF union only (`3000` images). Restoration outputs were ignored; no training, no prediction images, no locked-test enumeration.

## Summary

- R mean: `0.491335`
- R std mean: `0.081058` (gate `>= 0.10`)
- corr(R, input-GT L1): `-0.438983` (gate `> 0.10`)
- corr(R, A0 low+high error proxy): `-0.421714` (gate `> 0.10`)
- corr(R, dark-channel mean): `-0.884114` (direction gate `> 0`)
- heavy haze q4 vs q1 relative response: `-0.059567` (gate `>= 0.10`)
- A0-error q4 vs q1 relative response: `-0.066312` (gate `>= 0.10`)
- low-saturation q1 minus high-saturation q4 R mean: `-0.022944` (gate `>= 0`)

## Failed Gates

- `R_1_2_std_mean`
- `corr_R_input_gt_l1`
- `corr_R_a0_error_proxy`
- `corr_R_dark_channel_direction`
- `heavy_haze_q4_gt_q1_by_10pct`
- `a0_error_q4_gt_q1_by_10pct`
- `low_saturation_no_reverse`

## Proxy Direction Snapshot

| Proxy | q1 R mean | q4 R mean | corr(R, proxy) |
| --- | ---: | ---: | ---: |
| `input_gt_l1` | q1 `0.504377` | q4 `0.474333` | corr `-0.438983` |
| `a0_error_proxy_low_plus_high_l1` | q1 `0.511378` | q4 `0.477468` | corr `-0.421714` |
| `input_dark_channel_mean` | q1 `0.523416` | q4 `0.464575` | corr `-0.884114` |
| `input_saturation_proxy` | q1 `0.481255` | q4 `0.504199` | corr `0.356445` |
| `a0_psnr` | q1 `0.477588` | q4 `0.511271` | corr `0.504380` |

## Interpretation

This audit tests whether the existing v4.5 SDC-Lite response field behaves like a usable haze/error controller. It does not test restoration quality and does not authorize connecting R to skip/FAM/restoration outputs.
