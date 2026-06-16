# Haze4K v2.5 C13-F Diagnostic Decision

Decision: `C13_F_DIAGNOSTIC_COMPLETE_GATE_VS_RESIDUAL_DIRECTION_REVIEW`

Locked Haze4K test remained untouched.

## F0 Full-600 Leaderboard

| Variant | mean | hard | easy | positive | severe/600 | dSSIM | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| wd0375_teacher | 2.512202 | 3.505615 | 1.189484 | 0.973333 | 11.00 | 0.00167334 | True |
| c13a4_scale050 | 0.361713 | 0.564971 | 0.119759 | 0.696667 | 115.00 | 0.00026006 | False |
| c13a2_directzero256 | 0.356382 | 0.557847 | 0.108048 | 0.685000 | 124.00 | 0.00026261 | False |
| a5_a4sweep_s030 | 0.253058 | 0.343011 | 0.155960 | 0.743333 | 57.00 | 0.00018498 | False |
| a5_a4sweep_s025 | 0.220108 | 0.286678 | 0.153672 | 0.758333 | 42.00 | 0.00016245 | False |
| c13a3_adaptive050 | 0.044723 | 0.025461 | 0.091248 | 0.800000 | 0.00 | 0.00005378 | False |

## Oracle Summary

- `actual_scale`: mean `0.361713`, hard `0.564971`, easy `0.119759`, positive `0.696667`, severe/600 `115.00`, pass `False`
- `band_independent_oracle`: mean `0.554932`, hard `0.730825`, easy `0.338570`, positive `0.983333`, severe/600 `0.00`, pass `True`
- `hh_only_oracle`: mean `0.023067`, hard `0.001285`, easy `0.075492`, positive `0.916667`, severe/600 `0.00`, pass `False`
- `high_only_oracle`: mean `0.023268`, hard `0.001405`, easy `0.075645`, positive `0.968333`, severe/600 `0.00`, pass `False`
- `hl_only_oracle`: mean `0.023191`, hard `0.001361`, easy `0.075583`, positive `0.965000`, severe/600 `0.00`, pass `False`
- `lh_only_oracle`: mean `0.023158`, hard `0.001333`, easy `0.075583`, positive `0.963333`, severe/600 `0.00`, pass `False`
- `ll_only_oracle`: mean `0.554681`, hard `0.730671`, easy `0.338372`, positive `0.961667`, severe/600 `0.00`, pass `True`
- `patch_scale_oracle`: mean `0.750215`, hard `0.818064`, easy `0.624435`, positive `1.000000`, severe/600 `0.00`, pass `True`
- `per_image_scale_oracle`: mean `0.554817`, hard `0.730784`, easy `0.338369`, positive `0.961667`, severe/600 `0.00`, pass `True`

## A0 Reference

- count: `600`
- mean A0 PSNR: `36.837436`
- mean A0 SSIM: `0.994519`

## Teacher Reference

- count: `600`
- mean teacher dPSNR: `2.512202`
- mean teacher dSSIM: `0.001673`
