# v2h-A D7c Risk-Coverage Calibration

Status: `COMPLETED_GATE_PASS`

Decision label: `V2H_A_D7C_RISK_COVERAGE_PASS_AUTHORIZE_SHADOW`

Policy: train-calib selects thresholds; val-inner is report-only. No locked test, D2, F5, v3, RARM, adapter training, or new probe training was run.

## Primary Operating Point

| Score | Coverage | Action recall | Low-adj recall | Negative false | Negative false p95 | Isolated hit | Density action recall | Density negative false |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D7c fixed | 0.302695 | 0.548312 | 0.155904 | 0.002974 | 0.047619 | 0.022366 | 0.448391 | 0.047786 |

## Decision

v2h-B shadow-modulation only
