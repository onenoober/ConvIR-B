# v2g G2b Oracle-Gain Diagnostic Summary

Status: `COMPLETED_G2B_ORACLE_GAIN_DIAGNOSTIC`

Policy: val_inner only; locked Haze4K test, D2, RARM, v3, and F5 were not run.

Path contract: `g2b_oracle_path_contract.json`.

## Key Rows

| Region | Coverage | Removed residual energy | Oracle PSNR gain | Trans mean | Veil mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_ldhn | 0.080287 | 0.150716 | 0.709469 | 0.474674 | 0.525326 |
| ldhn_adjacent_to_haze | 0.009481 | 0.014645 | 0.064071 | 0.347740 | 0.652260 |
| ldhn_isolated | 0.070806 | 0.136071 | 0.635220 | 0.491670 | 0.508330 |
| missed_ldhn | 0.077135 | 0.136579 | 0.637772 | 0.476151 | 0.523849 |
| missed_adjacent_ldhn | 0.008000 | 0.011130 | 0.048609 | 0.348766 | 0.651234 |
| missed_isolated_ldhn | 0.069135 | 0.125448 | 0.582146 | 0.490892 | 0.509108 |
| ldln_false_tail_pred_high | 0.000415 | 0.000024 | 0.000105 | 0.400260 | 0.599740 |
| low_density_low_need | 0.153845 | 0.005035 | 0.021921 | 0.678320 | 0.321680 |

## Interpretation

These rows separate action value from haze actionability. A region can remove residual energy yet still have high transmission/low veil, which means it should not automatically become an RARM-positive target.
