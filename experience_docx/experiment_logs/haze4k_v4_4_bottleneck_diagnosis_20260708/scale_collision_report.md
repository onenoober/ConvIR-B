# v4.4 Scale Collision Report

Route id: `haze4k_v4_4_bottleneck_diagnosis_20260708`

Locked test touched: `false`

## Primary Internal256 Result

- A1 mean delta PSNR: `0.060228`
- A2 mean delta PSNR: `0.066864`
- A3 mean delta PSNR: `0.028960`
- Expected additive mean: `0.127093`
- Mean interaction delta: `-0.098133`
- A3 positive ratio: `0.484375`
- Both A1/A2 positive count: `131`
- A3 negative given both A1/A2 positive: `0.21374045801526717`

## Legacy Trainfit128 Check

- A3 mean delta PSNR: `-0.045744`
- Mean interaction delta: `-0.094506`

## Module Read

- A1 SDFM_1_2 R_std mean: `0.286745`
- A3 SDFM_1_2 R_std mean: `0.090339`
- A2 GST_1_2 effective update mean: `0.00352033`
- A3 GST_1_2 effective update mean: `0.00319296`

Interpretation: negative interaction on `internal_holdout256` supports the after-A3 bottleneck diagnosis. If A3 is also negative when A1 and A2 are both positive, the strongest hypothesis is same-scale intervention collision rather than insufficient training.
