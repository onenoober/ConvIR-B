# Haze4K APDR-v0.4E E0 Fixed-Code Rerun

Date: 2026-06-04

Status: E0 rerun completed on AutoDL from clean `826caaf`, but exact E0 numeric
seal remains blocked by mapper-name compatibility for Rule A.

## Result

`v04e_locked_threshold_confirm_summary.json` reports:

| Rule | Status | Keep | Mean gain | Hard gain | Easy gain | Strong/severe | Oracle recovery | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Rule A `global_plus_spatial_kenel_knn_9` | missing candidate | - | - | - | - | - | - | fail |
| Rule B `spatial_priors_ridge_10` | present | `45/128` | `+0.2141 dB` | `+0.4528 dB` | `+0.0625 dB` | `1/0` | `0.2363` | pass |

Rule B matches the previous E0 values to numerical precision. Rule A is missing
because `826caaf` still filtered `kernel_knn` outputs with historical
`kenel_knn` candidate names. Current code has been patched with mapper alias
compatibility, but this directory records the pre-alias-fix clean rerun.

## Decision

E0 alone authorizes only OOF calibration. Because E1 fails, no full router,
local correction, dense residual head, E2, or stop20 is authorized from this
route.
