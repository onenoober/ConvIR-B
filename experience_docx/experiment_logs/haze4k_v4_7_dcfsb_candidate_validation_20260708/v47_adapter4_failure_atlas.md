# Haze4K v4.7 Adapter4 Failure Atlas

Route id: `haze4k_v4_7_dcfsb_candidate_validation_20260708`

Split: `internal_holdout256` train-derived holdout only. Locked test touched/enumerated: `false` / `false`.

## Gate Summary

- mean dPSNR: `0.044404`
- positive ratio: `0.625000`
- p5 dPSNR: `-0.216141`
- mean dHighL1: `-0.0000005702`
- bootstrap 95% CI: `[0.024481, 0.065079]`
- sign-test one-sided p: `3.802649e-05`
- severe counts: `{'delta_lt_minus_0_50': 0, 'delta_lt_minus_0_25': 10, 'delta_lt_minus_0_10': 44}`
- systematic failure flags: 0

## Worst32 Proxy Concentration

| Proxy | q1 | q2 | q3 | q4 | max ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `input_gt_l1` | 11 | 9 | 7 | 5 | 0.344 |
| `a0_error_proxy_low_plus_high_l1` | 7 | 10 | 6 | 9 | 0.312 |
| `a0_psnr` | 10 | 6 | 10 | 6 | 0.312 |
| `input_dark_channel_mean` | 11 | 8 | 8 | 5 | 0.344 |
| `input_brightness_mean` | 11 | 9 | 7 | 5 | 0.344 |
| `input_saturation_proxy` | 10 | 7 | 9 | 6 | 0.312 |
| `gt_texture_proxy` | 7 | 11 | 7 | 7 | 0.344 |
| `hazy_texture_proxy` | 7 | 9 | 11 | 5 | 0.344 |
| `high_gate_std` | 9 | 6 | 11 | 6 | 0.344 |
| `high_low_energy_ratio` | 8 | 5 | 12 | 7 | 0.375 |

Systematic flags:

None.

## Worst 16 Images

| Rank | Image | dPSNR | A0 PSNR | input-GT L1 | A0 proxy L1 | dark channel | GT texture |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `2065_0.51_1.6.png` | -0.415844 | 41.898407 | 0.086515 | 0.006931 | 0.329654 | 0.025062 |
| 2 | `2571_0.85_1.0.png` | -0.350124 | 47.151886 | 0.132478 | 0.003967 | 0.338162 | 0.023464 |
| 3 | `418_0.66_0.66.png` | -0.344364 | 40.122486 | 0.064129 | 0.008015 | 0.471115 | 0.044929 |
| 4 | `2404_0.65_1.05.png` | -0.311901 | 41.888905 | 0.031595 | 0.006536 | 0.556296 | 0.034272 |
| 5 | `126_0.75_1.47.png` | -0.295771 | 28.263138 | 0.134615 | 0.023094 | 0.603664 | 0.058089 |
| 6 | `1969_0.52_1.0.png` | -0.290764 | 41.916725 | 0.078468 | 0.007718 | 0.432259 | 0.034606 |
| 7 | `2428_0.56_1.84.png` | -0.288345 | 42.568108 | 0.085392 | 0.006859 | 0.524758 | 0.033770 |
| 8 | `2238_0.81_1.38.png` | -0.281029 | 44.299072 | 0.135212 | 0.005223 | 0.372903 | 0.035192 |
| 9 | `402_0.54_1.36.png` | -0.278343 | 39.900555 | 0.091116 | 0.009032 | 0.490213 | 0.051634 |
| 10 | `2402_0.8_0.8.png` | -0.255051 | 40.857361 | 0.027451 | 0.007019 | 0.569433 | 0.034272 |
| 11 | `545_0.7_0.92.png` | -0.239479 | 37.690948 | 0.060837 | 0.011375 | 0.471057 | 0.058774 |
| 12 | `1454_0.62_1.48.png` | -0.227699 | 33.793892 | 0.145701 | 0.017345 | 0.629628 | 0.035393 |
| 13 | `111_0.57_1.22.png` | -0.220364 | 38.626282 | 0.156184 | 0.010987 | 0.591303 | 0.049917 |
| 14 | `1288_0.95_1.97.png` | -0.214733 | 31.967030 | 0.217317 | 0.023246 | 0.766119 | 0.024460 |
| 15 | `1308_0.84_1.04.png` | -0.205856 | 35.422279 | 0.116187 | 0.016281 | 0.566447 | 0.041278 |
| 16 | `2027_0.69_0.59.png` | -0.200165 | 43.080868 | 0.065126 | 0.006175 | 0.314890 | 0.022986 |

## Interpretation

This atlas is a compact text audit, not a visual inspection set. It checks whether adapter4's worst internal_holdout regressions concentrate in a single proxy quartile strongly enough to suggest a systematic failure mode. The gate treats a bin as systematic only when worst32 concentration is high and the whole bin has negative mean movement and low positive ratio.
