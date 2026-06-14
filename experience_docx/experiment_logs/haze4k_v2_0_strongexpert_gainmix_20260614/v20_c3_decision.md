# Haze4K v2.0 C3 Train-Only Shifted Validation

Decision: `C3_SHIFTED_VALIDATION_PASS_START_FORMAL_5X3`

C3 validates the C2d alpha-shrink policy family by holding out train-derived bins and selecting only the scalar output-diff threshold on the remaining bins.
The locked Haze4K test remains untouched.

## Dimension Summary

| Dimension | Pass | Mean | Hard | Easy | dSSIM | Severe/600 | Min Bin Mean | Max Bin Severe/600 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `split` | `True` | `0.331992` | `0.253350` | `0.476962` | `0.00023533` | `38.0` | `0.287334` | `38.0` |
| `airlight_q4` | `True` | `0.332107` | `0.260358` | `0.468415` | `0.00024210` | `35.0` | `0.264858` | `68.0` |
| `beta_haze_q4` | `True` | `0.333458` | `0.260494` | `0.476918` | `0.00024035` | `40.0` | `0.237904` | `44.0` |
| `depth_mean_q4` | `True` | `0.323090` | `0.253883` | `0.457586` | `0.00022909` | `33.0` | `0.301593` | `40.0` |
| `input_grad_lowtexture_q4` | `True` | `0.331300` | `0.261183` | `0.469822` | `0.00024124` | `37.0` | `0.280106` | `64.0` |
| `input_dark_q4` | `True` | `0.328953` | `0.254661` | `0.468065` | `0.00023544` | `38.0` | `0.239970` | `56.0` |
| `diff_abs_q4` | `True` | `0.332612` | `0.254988` | `0.477789` | `0.00023825` | `39.0` | `0.263139` | `44.0` |
| `a0_psnr_stress_q4` | `True` | `0.333593` | `0.258173` | `0.476962` | `0.00023968` | `38.0` | `0.258173` | `64.0` |

## Interpretation

- Formal 5x3 replay is authorized only if every dimension passes.
- Locked test remains blocked until formal 5x3 also passes and the final policy is sealed.
